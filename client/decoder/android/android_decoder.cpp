/*
 * WiVRn VR streaming
 * Copyright (C) 2022  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2022  Patrick Nicolas <patricknicolas@laposte.net>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "android_decoder.h"
#include "application.h"
#include "scenes/stream.h"
#include "utils/named_thread.h"
#include <android/hardware_buffer.h>
#include <cassert>
#include <magic_enum.hpp>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#include <mutex>
#include <queue>
#include <ranges>
#include <semaphore>
#include <spdlog/spdlog.h>
#include <thread>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_android.h>
#include <vulkan/vulkan_raii.hpp>

DEUGLIFY(AMediaFormat)

struct wivrn::android::decoder::mapped_hardware_buffer
{
	vk::raii::DeviceMemory memory = nullptr;
	vk::raii::Image vimage = nullptr;
	vk::raii::ImageView image_view = nullptr;
	vk::ImageLayout layout = vk::ImageLayout::eUndefined;
};

namespace
{
const char * mime(wivrn::video_codec codec)
{
	using c = wivrn::video_codec;
	switch (codec)
	{
		case c::h264:
			return "video/avc";
		case c::h265:
			return "video/hevc";
		case c::av1:
			return "video/av01";
	}
	assert(false);
}

void check(media_status_t status, const char * msg)
{
	if (status != AMEDIA_OK)
	{
		spdlog::error("{}: MediaCodec error {}", msg, (int)status);
		throw std::runtime_error("MediaCodec error");
	}
}

} // namespace

namespace wivrn::android
{

class media_codec_access
{
	struct input_job
	{
		decoder * decoder;
		decoder::frame_info info;
		size_t buffer_idx;
		size_t data_size;

		struct compare
		{
			bool operator()(const input_job & a, const input_job & b)
			{
				return a.info.feedback.frame_index > b.info.feedback.frame_index;
			}
		};
	};
	std::mutex mutex;
	std::condition_variable cv;
	bool exit_request = false;
	std::binary_semaphore image_ready{0};
	std::priority_queue<input_job, std::vector<input_job>, input_job::compare> input_jobs;
	std::thread worker;

	media_codec_access();

public:
	~media_codec_access();
	static std::shared_ptr<media_codec_access> get();
	void stop(const decoder &);

	// schedule a call to AMediaCodec_queueInputBuffer
	void push_input(
	        decoder *,
	        decoder::frame_info info,
	        size_t buffer_idx,
	        size_t data_size);

	void on_image_ready()
	{
		spdlog::info("image ready");
		image_ready.release();
	}
};

media_codec_access::media_codec_access()
{
	worker = utils::named_thread(
	        "decoder",
	        [this]() {
		        while (true)
		        {
			        std::unique_lock lock(mutex);
			        if (exit_request)
				        return;
			        if (input_jobs.empty())
				        cv.wait(lock);
			        if (not input_jobs.empty())
			        {
				        auto job = input_jobs.top();
				        input_jobs.pop();
				        lock.unlock();

				        auto media_codec = job.decoder->media_codec.get();

				        job.info.feedback.sent_to_decoder = application::now();
				        auto status = AMediaCodec_queueInputBuffer(
				                media_codec,
				                job.buffer_idx,
				                0,
				                job.data_size,
				                job.info.feedback.frame_index * 10'000,
				                0);
				        if (status != AMEDIA_OK)
				        {
					        spdlog::error("AMediaCodec_queueInputBuffer: MediaCodec error {}({})",
					                      int(status),
					                      std::string(magic_enum::enum_name(status)).c_str());
					        continue;
				        }

				        ssize_t decoded = -1;
				        while (decoded == -1)
				        {
					        AMediaCodecBufferInfo info{};
					        decoded = AMediaCodec_dequeueOutputBuffer(media_codec, &info, -1);

					        // discard buffers without data (format changed)
					        if (info.size == 0)
					        {
						        AMediaCodec_releaseOutputBuffer(media_codec, decoded, false);
						        decoded = -1;
					        }
				        }

				        status = AMediaCodec_releaseOutputBuffer(media_codec, decoded, true);
				        if (status != AMEDIA_OK)
				        {
					        spdlog::error("AMediaCodec_releaseOutputBuffer: MediaCodec error {}({})",
					                      int(status),
					                      std::string(magic_enum::enum_name(status)).c_str());
					        continue;
				        }

				        // Wait for image to be rendered
				        image_ready.acquire();
				        if (exit_request)
					        return;

				        AImage_ptr image;
				        AImage * tmp;
				        status = AImageReader_acquireNextImage(job.decoder->image_reader.get(), &tmp);
				        if (status != AMEDIA_OK)
				        {
					        spdlog::error("AImageReader_acquireNextImage: MediaCodec error {}({})",
					                      int(status),
					                      std::string(magic_enum::enum_name(status)).c_str());
					        continue;
				        }
				        image.reset(tmp);

#ifndef NDEBUG
				        int64_t ts;
				        AImage_getTimestamp(tmp, &ts);
				        ts = (ts + 5'000'000) / 10'000'000;
				        if (ts != job.info.feedback.frame_index)
				        {
					        spdlog::error("invalid frame index, got {} expected {}", ts, job.info.feedback.frame_index);
				        }
#endif

				        job.decoder->on_image_available(std::move(image), job.info);
			        }
		        }
	        });
}

media_codec_access::~media_codec_access()
{
	{
		std::unique_lock lock(mutex);
		input_jobs = decltype(input_jobs)();
		exit_request = true;
		cv.notify_all();
		image_ready.release();
	}
	worker.join();
}

std::shared_ptr<media_codec_access> media_codec_access::get()
{
	static std::weak_ptr<media_codec_access> instance;
	static std::mutex m;
	std::unique_lock lock(m);
	auto s = instance.lock();
	if (s)
		return s;
	s.reset(new media_codec_access);
	instance = s;
	return s;
}

void media_codec_access::stop(const decoder & decoder)
{
	std::unique_lock lock(mutex);
	decltype(input_jobs) jobs;
	for (; not input_jobs.empty(); input_jobs.pop())
	{
		const auto & top = input_jobs.top();
		if (top.decoder != &decoder)
			jobs.push(top);
	}
	input_jobs = std::move(jobs);
}

void media_codec_access::push_input(
        decoder * decoder,
        decoder::frame_info info,
        size_t buffer_idx,
        size_t data_size)
{
	std::unique_lock lock(mutex);
	input_jobs.emplace(decoder, info, buffer_idx, data_size);
	cv.notify_all();
}

decoder::decoder(
        vk::raii::Device & device,
        vk::raii::PhysicalDevice & physical_device,
        const wivrn::to_headset::video_stream_description::item & description,
        float fps,
        uint8_t stream_index,
        std::weak_ptr<scenes::stream> weak_scene,
        shard_accumulator * accumulator) :
        jobs(media_codec_access::get()),
        description(description),
        stream_index(stream_index),
        fps(fps),
        device(device),
        weak_scene(weak_scene),
        accumulator(accumulator)
{
	spdlog::info("hbm_mutex.native_handle() = {}", (void *)hbm_mutex.native_handle());

	AImageReader * ir;
	check(AImageReader_newWithUsage(
	              description.video_width,
	              description.video_height,
	              AIMAGE_FORMAT_PRIVATE,
	              AHARDWAREBUFFER_USAGE_CPU_READ_NEVER | AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
	              scenes::stream::image_buffer_size + 4 /* maxImages */,
	              &ir),
	      "AImageReader_newWithUsage");
	image_reader.reset(ir, AImageReader_deleter{});

	vkGetAndroidHardwareBufferPropertiesANDROID =
	        application::get_vulkan_proc<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
	                "vkGetAndroidHardwareBufferPropertiesANDROID");
	AMediaFormat_ptr format(AMediaFormat_new());
	AMediaFormat_setString(format.get(), AMEDIAFORMAT_KEY_MIME, mime(description.codec));
	// AMediaFormat_setInt32(format.get(), "vendor.qti-ext-dec-low-latency.enable", 1); // Qualcomm low
	// latency mode
	AMediaFormat_setInt32(format.get(), AMEDIAFORMAT_KEY_WIDTH, description.video_width);
	AMediaFormat_setInt32(format.get(), AMEDIAFORMAT_KEY_HEIGHT, description.video_height);
	AMediaFormat_setInt32(format.get(), AMEDIAFORMAT_KEY_OPERATING_RATE, std::ceil(fps));
	AMediaFormat_setInt32(format.get(), AMEDIAFORMAT_KEY_PRIORITY, 0);

	media_codec.reset(AMediaCodec_createDecoderByType(mime(description.codec)));

	if (not media_codec)
		throw std::runtime_error(std::string("Cannot create decoder for MIME type ") + mime(description.codec));

	AImageReader_ImageListener listener{
	        .context = jobs.get(),
	        .onImageAvailable = [](void * context, AImageReader * reader) {
		        ((media_codec_access *)(context))->on_image_ready();
	        }};

	AImageReader_setImageListener(image_reader.get(), &listener);

	char * codec_name;
	check(AMediaCodec_getName(media_codec.get(), &codec_name), "AMediaCodec_getName");
	spdlog::info("Created MediaCodec decoder \"{}\"", codec_name);
	AMediaCodec_releaseName(media_codec.get(), codec_name);

	ANativeWindow * window;

	check(AImageReader_getWindow(image_reader.get(), &window), "AImageReader_getWindow");

	check(AMediaCodec_configure(media_codec.get(), format.get(), window, nullptr /* crypto */, 0 /* flags */),
	      "AMediaCodec_configure");

	check(AMediaCodec_start(media_codec.get()), "AMediaCodec_start");
}

decoder::~decoder()
{
	if (media_codec)
	{
		AMediaCodec_stop(media_codec.get());
		jobs->stop(*this);
	}
	spdlog::info("decoder::~decoder");
}

void decoder::push_data(std::span<std::span<const uint8_t>> data, uint64_t frame_index, bool partial)
{
	if (current_input_buffer.data == nullptr)
	{
		current_input_buffer.idx = AMediaCodec_dequeueInputBuffer(media_codec.get(), -1);
		current_input_buffer.data = AMediaCodec_getInputBuffer(media_codec.get(),
		                                                       current_input_buffer.idx,
		                                                       &current_input_buffer.capacity);
		current_input_buffer.data_size = 0;
	}
	else if (current_input_buffer.frame_index != frame_index)
	{
		// Reuse the input buffer, discard existing data
		current_input_buffer.data_size = 0;
	}
	current_input_buffer.frame_index = frame_index;

	for (const auto & sub_data: data)
	{
		if (current_input_buffer.data_size + sub_data.size() > current_input_buffer.capacity)
		{
			spdlog::error("data to decode is larger than decoder buffer, skipping frame");
			return;
		}

		memcpy(current_input_buffer.data + current_input_buffer.data_size, sub_data.data(), sub_data.size());
		current_input_buffer.data_size += sub_data.size();
	}
}

void decoder::frame_completed(wivrn::from_headset::feedback & feedback, const wivrn::to_headset::video_stream_data_shard::timing_info_t & timing_info, const wivrn::to_headset::video_stream_data_shard::view_info_t & view_info)
{
	if (not media_codec)
	{
		// If media_codec is not initialized, frame processing ends here
		if (auto scene = weak_scene.lock())
			scene->send_feedback(feedback);
	}

	auto f = std::make_shared<wivrn::from_headset::feedback>(feedback);

	jobs->push_input(
	        this,
	        frame_info{
	                .feedback = feedback,
	                .timing_info = timing_info,
	                .view_info = view_info,
	        },
	        current_input_buffer.idx,
	        current_input_buffer.data_size);
	current_input_buffer = input_buffer{};
}

void decoder::on_image_available(AImage_ptr aimage, frame_info info)
{
	try
	{
		auto vk_data = map_hardware_buffer(aimage.get());

		auto handle = std::make_shared<decoder::blit_handle>(
		        info.feedback,
		        info.timing_info,
		        info.view_info,
		        vk_data->image_view,
		        *vk_data->vimage,
		        &vk_data->layout,
		        vk_data,
		        image_reader,
		        std::move(aimage));

		if (auto scene = weak_scene.lock())
			scene->push_blit_handle(accumulator, std::move(handle));
	}
	catch (...)
	{
	}
}

void decoder::create_sampler(const AHardwareBuffer_Desc & buffer_desc, vk::AndroidHardwareBufferFormatPropertiesANDROID & ahb_format)
{
	assert(ahb_format.externalFormat != 0);
	spdlog::info("AndroidHardwareBufferProperties");
	spdlog::info("  Vulkan format: {}", vk::to_string(ahb_format.format));
	spdlog::info("  External format: {:#x}", ahb_format.externalFormat);
	spdlog::info("  Format features: {}", vk::to_string(ahb_format.formatFeatures));
	spdlog::info("  samplerYcbcrConversionComponents: ({}, {}, {}, {})",
	             vk::to_string(ahb_format.samplerYcbcrConversionComponents.r),
	             vk::to_string(ahb_format.samplerYcbcrConversionComponents.g),
	             vk::to_string(ahb_format.samplerYcbcrConversionComponents.b),
	             vk::to_string(ahb_format.samplerYcbcrConversionComponents.a));
	spdlog::info("  Suggested YCbCr model: {}", vk::to_string(ahb_format.suggestedYcbcrModel));
	spdlog::info("  Suggested YCbCr range: {}", vk::to_string(ahb_format.suggestedYcbcrRange));
	spdlog::info("  Suggested X chroma offset: {}", vk::to_string(ahb_format.suggestedXChromaOffset));
	spdlog::info("  Suggested Y chroma offset: {}", vk::to_string(ahb_format.suggestedYChromaOffset));

	vk::Filter yuv_filter;
	if (ahb_format.formatFeatures & vk::FormatFeatureFlagBits::eSampledImageYcbcrConversionLinearFilter)
		yuv_filter = vk::Filter::eLinear;
	else
		yuv_filter = vk::Filter::eNearest;

	// Create VkSamplerYcbcrConversion
	vk::StructureChain ycbcr_create_info{
	        vk::SamplerYcbcrConversionCreateInfo{
	                .format = vk::Format::eUndefined,
	                .ycbcrModel = ahb_format.suggestedYcbcrModel,
	                .ycbcrRange = ahb_format.suggestedYcbcrRange,
	                .components = ahb_format.samplerYcbcrConversionComponents,
	                .xChromaOffset = ahb_format.suggestedXChromaOffset,
	                .yChromaOffset = ahb_format.suggestedYChromaOffset,
	                .chromaFilter = yuv_filter,
	        },
	        vk::ExternalFormatANDROID{
	                .externalFormat = ahb_format.externalFormat,
	        },
	};

	// suggested values from decoder don't actually read the metadata, so it's garbage
	if (description.range)
		ycbcr_create_info.get<vk::SamplerYcbcrConversionCreateInfo>().ycbcrRange = vk::SamplerYcbcrRange(*description.range);

	if (description.color_model)
		ycbcr_create_info.get<vk::SamplerYcbcrConversionCreateInfo>().ycbcrModel = vk::SamplerYcbcrModelConversion(*description.color_model);

	ycbcr_conversion = vk::raii::SamplerYcbcrConversion(device, ycbcr_create_info.get<vk::SamplerYcbcrConversionCreateInfo>());

	// Create VkSampler
	vk::StructureChain sampler_info{
	        vk::SamplerCreateInfo{
	                .magFilter = yuv_filter,
	                .minFilter = yuv_filter,
	                .mipmapMode = vk::SamplerMipmapMode::eNearest,
	                .addressModeU = vk::SamplerAddressMode::eClampToEdge,
	                .addressModeV = vk::SamplerAddressMode::eClampToEdge,
	                .addressModeW = vk::SamplerAddressMode::eClampToEdge,
	                .mipLodBias = 0.0f,
	                .anisotropyEnable = VK_FALSE,
	                .maxAnisotropy = 1,
	                .compareEnable = VK_FALSE,
	                .compareOp = vk::CompareOp::eNever,
	                .minLod = 0.0f,
	                .maxLod = 0.0f,
	                .borderColor = vk::BorderColor::eFloatOpaqueWhite, // TODO TBC
	                .unnormalizedCoordinates = VK_FALSE,
	        },
	        vk::SamplerYcbcrConversionInfo{
	                .conversion = *ycbcr_conversion,
	        },
	};

	ycbcr_sampler = vk::raii::Sampler(device, sampler_info.get());
}

std::shared_ptr<decoder::mapped_hardware_buffer> decoder::map_hardware_buffer(AImage * image)
{
	AHardwareBuffer * hardware_buffer;
	check(AImage_getHardwareBuffer(image, &hardware_buffer), "AImage_getHardwareBuffer");

	std::unique_lock lock(hbm_mutex);

	AHardwareBuffer_Desc buffer_desc{};
	AHardwareBuffer_describe(hardware_buffer, &buffer_desc);

	auto [properties, format_properties] = device.getAndroidHardwareBufferPropertiesANDROID<vk::AndroidHardwareBufferPropertiesANDROID, vk::AndroidHardwareBufferFormatPropertiesANDROID>(*hardware_buffer);

	if (!*ycbcr_sampler || memcmp(&ahb_format, &format_properties, sizeof(format_properties)))
	{
		memcpy(&ahb_format, &format_properties, sizeof(format_properties));
		extent = {buffer_desc.width, buffer_desc.height};
		create_sampler(buffer_desc, ahb_format);
		hardware_buffer_map.clear();
		// TODO tell the reprojector to recreate the pipeline
	}

	auto it = hardware_buffer_map.find(hardware_buffer);
	if (it != hardware_buffer_map.end())
		return it->second;

	vk::StructureChain img_info{
	        vk::ImageCreateInfo{
	                .flags = {},
	                .imageType = vk::ImageType::e2D,
	                .format = vk::Format::eUndefined,
	                .extent = {buffer_desc.width, buffer_desc.height, 1},
	                .mipLevels = 1,
	                .arrayLayers = 1,
	                .samples = vk::SampleCountFlagBits::e1,
	                .tiling = vk::ImageTiling::eOptimal,
	                .usage = vk::ImageUsageFlagBits::eSampled,
	                .sharingMode = vk::SharingMode::eExclusive,
	                .initialLayout = vk::ImageLayout::eUndefined,
	        },
	        vk::ExternalMemoryImageCreateInfo{
	                .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eAndroidHardwareBufferANDROID,
	        },
	        vk::ExternalFormatANDROID{
	                .externalFormat = format_properties.externalFormat,
	        },
	};

	vk::raii::Image vimage(device, img_info.get());

	assert(properties.memoryTypeBits != 0);
	vk::StructureChain mem_info{
	        vk::MemoryAllocateInfo{
	                .allocationSize = properties.allocationSize,
	                .memoryTypeIndex = (uint32_t)(ffs(properties.memoryTypeBits) - 1),
	        },
	        vk::MemoryDedicatedAllocateInfo{
	                .image = *vimage,
	        },
	        vk::ImportAndroidHardwareBufferInfoANDROID{
	                .buffer = hardware_buffer,
	        },
	};

	vk::raii::DeviceMemory memory(device, mem_info.get());

	vimage.bindMemory(*memory, 0);

	vk::StructureChain iv_info{
	        vk::ImageViewCreateInfo{
	                .image = *vimage,
	                .viewType = vk::ImageViewType::e2D,
	                .format = vk::Format::eUndefined,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .baseMipLevel = 0,
	                        .levelCount = 1,
	                        .baseArrayLayer = 0,
	                        .layerCount = 1,
	                },
	        },
	        vk::SamplerYcbcrConversionInfo{
	                .conversion = *ycbcr_conversion,
	        },
	};

	application::ignore_debug_reports_for(*vimage);
	vk::raii::ImageView image_view(device, iv_info.get());
	application::unignore_debug_reports_for(*vimage);

	auto handle = std::make_shared<mapped_hardware_buffer>();
	handle->vimage = std::move(vimage);
	handle->image_view = std::move(image_view);
	handle->memory = std::move(memory);

	hardware_buffer_map[hardware_buffer] = handle;
	return handle;
}

static bool hardware_accelerated(AMediaCodec * media_codec)
{
	// MediaCodecInfo has isHardwareAccelerated, but this does not exist in NDK.
	char * name;
	AMediaCodec_getName(media_codec, &name);
	auto release = [&]() {
		AMediaCodec_releaseName(media_codec, name);
	};
	for (const char * prefix: {
	             "OMX.google",
	             "c2.android",
	     })
	{
		if (std::string_view(name).starts_with(prefix))
		{
			release();
			return false;
		}
	}
	release();
	return true;
}

std::vector<wivrn::video_codec> decoder::supported_codecs()
{
	std::vector<wivrn::video_codec> result;
	// Make sure we update this code when codecs are changed
	static_assert(magic_enum::enum_count<wivrn::video_codec>() == 3);

	// In order or preference, from preferred to least preferred
	for (auto codec: {
	             wivrn::video_codec::av1,
	             wivrn::video_codec::h264,
	             wivrn::video_codec::h265,
	     })
	{
		AMediaCodec_ptr media_codec(AMediaCodec_createDecoderByType(mime(codec)));

		bool supported = media_codec and hardware_accelerated(media_codec.get());
		if (supported)
			result.push_back(codec);

		spdlog::info("video codec {}: {}supported", magic_enum::enum_name(codec), supported ? "" : "NOT ");
	}

	return result;
}

} // namespace wivrn::android
