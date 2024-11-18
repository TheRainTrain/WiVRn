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

#pragma once

#include "wivrn_packets.h"
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>

#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#include <vulkan/vulkan_raii.hpp>

#define DEUGLIFY(x)                      \
	struct x##_deleter               \
	{                                \
		void operator()(x * ptr) \
		{                        \
			x##_delete(ptr); \
		}                        \
	};                               \
	using x##_ptr = std::unique_ptr<x, x##_deleter>;

DEUGLIFY(AImage)
DEUGLIFY(AImageReader)
DEUGLIFY(AMediaCodec)

namespace wivrn
{
class shard_accumulator;
}

namespace scenes
{
class stream;
}

namespace wivrn::android
{
class media_codec_access;
class decoder
{
	friend class media_codec_access;

public:
	struct mapped_hardware_buffer;

	struct blit_handle
	{
		wivrn::from_headset::feedback feedback;
		wivrn::to_headset::video_stream_data_shard::timing_info_t timing_info;
		wivrn::to_headset::video_stream_data_shard::view_info_t view_info;
		vk::raii::ImageView & image_view;
		vk::Image image = nullptr;
		vk::ImageLayout * current_layout = nullptr;

		std::shared_ptr<mapped_hardware_buffer> vk_data;

		std::shared_ptr<AImageReader> image_reader;
		AImage_ptr aimage;
	};

private:
	std::shared_ptr<media_codec_access> jobs;
	wivrn::to_headset::video_stream_description::item description;
	uint8_t stream_index;
	float fps;

	vk::raii::Device & device;

	vk::AndroidHardwareBufferFormatPropertiesANDROID ahb_format;
	vk::raii::SamplerYcbcrConversion ycbcr_conversion = nullptr;
	vk::raii::Sampler ycbcr_sampler = nullptr;
	vk::Extent2D extent{};

	std::mutex hbm_mutex;
	std::shared_ptr<AImageReader> image_reader;

	AMediaCodec_ptr media_codec;
	std::weak_ptr<scenes::stream> weak_scene;
	shard_accumulator * accumulator;

	PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID;

	struct frame_info
	{
		wivrn::from_headset::feedback feedback;
		wivrn::to_headset::video_stream_data_shard::timing_info_t timing_info;
		wivrn::to_headset::video_stream_data_shard::view_info_t view_info;
	};
	void on_image_available(AImage_ptr, frame_info);

	struct input_buffer
	{
		int32_t idx;
		uint64_t frame_index = 0;
		size_t data_size = 0;
		size_t capacity = 0;
		uint8_t * data = nullptr;
	};

	input_buffer current_input_buffer; // Only accessed in network thread

	std::unordered_map<AHardwareBuffer *, std::shared_ptr<mapped_hardware_buffer>> hardware_buffer_map;
	vk::raii::RenderPass renderpass = nullptr;

	void create_sampler(const AHardwareBuffer_Desc & buffer_desc, vk::AndroidHardwareBufferFormatPropertiesANDROID & ahb_format);
	std::shared_ptr<mapped_hardware_buffer> map_hardware_buffer(AImage *);

public:
	decoder(vk::raii::Device & device,
	        vk::raii::PhysicalDevice & physical_device,
	        const wivrn::to_headset::video_stream_description::item & description,
	        float fps,
	        uint8_t stream_index,
	        std::weak_ptr<scenes::stream> scene,
	        shard_accumulator * accumulator);

	decoder(const decoder &) = delete;
	decoder(decoder &&) = delete;
	~decoder();

	void push_data(std::span<std::span<const uint8_t>> data, uint64_t frame_index, bool partial);

	void frame_completed(
	        wivrn::from_headset::feedback & feedback,
	        const wivrn::to_headset::video_stream_data_shard::timing_info_t & timing_info,
	        const wivrn::to_headset::video_stream_data_shard::view_info_t & view_info);

	const auto & desc() const
	{
		return description;
	}

	vk::Sampler sampler()
	{
		return *ycbcr_sampler;
	}

	vk::Extent2D image_size()
	{
		return extent;
	}

	static std::vector<wivrn::video_codec> supported_codecs();
};

} // namespace wivrn::android
