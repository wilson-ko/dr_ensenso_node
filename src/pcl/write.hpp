#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <estd/result.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace dr {

/// Write a point cloud as binary PCD to an ostream.
template<typename PointT>
void writePcdBinary(std::ostream & stream, const pcl::PointCloud<PointT> &cloud);

/// Write a point cloud as binary PCD to a file.
template<typename PointT>
estd::result<void, estd::error> writePcdBinary(std::string const & path, const pcl::PointCloud<PointT> &cloud);

/// Write a point cloud as compressed binary PCD to an ostream.
template<typename PointT>
void writePcdBinaryCompressed(std::ostream & stream, const pcl::PointCloud<PointT> &cloud);

/// Write a point cloud as compressed binary PCD to a file.
template<typename PointT>
estd::result<void, estd::error> writePcdBinaryCompressed(std::string const & path, const pcl::PointCloud<PointT> &cloud);

namespace impl {
	/// Get the non-padding fields for a point cloud.
	template<typename PointT>
	std::vector<pcl::PCLPointField> fieldsWithoutPadding(pcl::PointCloud<PointT> const & cloud) {
		std::vector<pcl::PCLPointField> fields;
		pcl::getFields(cloud, fields);
		fields.erase(std::remove_if(fields.begin(), fields.end(), [] (pcl::PCLPointField const & field) { return field.name == "_"; }), fields.end());
		return fields;
	}

	/// Calculate the total size of the given fields.
	std::uint32_t totalFieldSize(std::vector<pcl::PCLPointField> const & fields);

	/// Calculate the offsets for transposed fields.
	std::vector<std::size_t> calculateTransposedFieldOffsets(std::vector<pcl::PCLPointField> const & fields, std::size_t points);

	// Convert the XYZRGBXYZRGB structure to XXYYZZRGBRGB to aid compression.
	template<typename PointT>
	std::tuple<std::unique_ptr<char[]>, std::uint32_t> transposeCloudFields(pcl::PointCloud<PointT> const & cloud) {
		// Get non-padding fields and allocate a memory block for the transposed data.
		std::vector<pcl::PCLPointField> fields = fieldsWithoutPadding(cloud);

		struct FieldInfo {
			std::uint32_t dest_offset;
			std::uint16_t size;
			std::uint16_t src_offset;
		};

		std::vector<FieldInfo> info;
		std::uint32_t total_size = 0;
		{
			info.reserve(fields.size());
			for (auto const & field : fields) {
				info.push_back(FieldInfo{total_size, std::uint16_t(pcl::getFieldSize(field.datatype)), std::uint16_t(field.offset)});
				total_size += info.back().size * cloud.size();
			}
		}

		auto data = std::make_unique<char[]>(total_size);
		char * dest = data.get();

		for (PointT const & point : cloud.points) {
			for (FieldInfo & info : info) {
				char const * src  = reinterpret_cast<char const *>(&point) + info.src_offset;
				std::memcpy(&dest[info.dest_offset], src, info.size);
				info.dest_offset += info.size;
			}
		}

		return std::make_tuple(std::move(data), total_size);
	}

	/// Open a file for writing.
	estd::result<std::ofstream, estd::error> openFileForWriting(std::string const & path);

	class MemoryMappedFileRw {
	protected:
		int fd_;
		void * data_;
		std::size_t length_;

		MemoryMappedFileRw(int fd, void * data, std::size_t length) : fd_{fd}, data_{data}, length_{length} {}

		/// Create a read/write memory map from an FD.
		static estd::result<MemoryMappedFileRw, estd::error> from_fd(int fd, std::size_t size);

	public:
		/// Unmap the memory.
		~MemoryMappedFileRw();

		// Disable copying.
		MemoryMappedFileRw(MemoryMappedFileRw const &) = delete;
		MemoryMappedFileRw & operator=(MemoryMappedFileRw const &) = delete;

		// Allow moving.
		MemoryMappedFileRw(MemoryMappedFileRw && other) noexcept;
		MemoryMappedFileRw & operator=(MemoryMappedFileRw && other) noexcept;

		/// Create a new file of the given size and open it as memory map.
		static estd::result<MemoryMappedFileRw, estd::error> create(std::string const & path, std::size_t size);

		/// Get the FD used for the mapping.
		int fd() const { return fd_; }

		/// Get the raw pointer to the mapped memory region.
		void * get() const { return data_; }

		/// Get the length in bytes of the mapped memory region.
		std::size_t length() const { return length_; }

		/// Get the mapped memory region as a specific pointer type.
		template<typename T> T * as() const { return reinterpret_cast<T *>(data_); }

		/// Truncate the mapped file.
		estd::result<void, estd::error> truncate(std::size_t size);
	};

}

template<typename PointT>
void writePcdBinary(std::ostream & stream, const pcl::PointCloud<PointT> &cloud) {
	// Write header.
	pcl::PCDWriter writer;
	stream << pcl::PCDWriter::generateHeader(cloud) << "DATA binary\n";

	// Write data.
	std::vector<pcl::PCLPointField> fields = impl::fieldsWithoutPadding(cloud);
	for (PointT const & point : cloud) {
		for (pcl::PCLPointField const & field : fields) {
			stream.write(reinterpret_cast<char const *>(&point) + field.offset, pcl::getFieldSize(field.datatype));
		}
	}
}

template<typename PointT>
estd::result<void, estd::error> writePcdBinary(std::string const & path, const pcl::PointCloud<PointT> &cloud) {
	// Calculate header an data size.
	std::string header = pcl::PCDWriter::generateHeader(cloud) + "DATA binary\n";
	std::vector<pcl::PCLPointField> fields = impl::fieldsWithoutPadding(cloud);
	std::uint32_t data_size = impl::totalFieldSize(fields) * cloud.size();
	std::size_t total_size = header.size() + data_size;

	// Create a new file of the right size and map it in memory.
	auto map = impl::MemoryMappedFileRw::create(path, total_size);
	if (!map) return map.error();

	// Write header.
	char * dest = map->as<char>();
	std::memcpy(dest, header.data(), header.size());
	dest += header.size();

	// Write data.
	for (PointT const & point : cloud) {
		for (pcl::PCLPointField const & field : fields) {
			int field_size = pcl::getFieldSize(field.datatype);
			std::memcpy(dest, reinterpret_cast<char const *>(&point) + field.offset, field_size);
			dest += field_size;
		}
	}

	return estd::in_place_valid;
}

template<typename PointT>
void writePcdBinaryCompressed(std::ostream & stream, const pcl::PointCloud<PointT> &cloud) {
	// Transpose the cloud fields so all values for the same field are contiguous.
	auto [data, size] = impl::transposeCloudFields(cloud);

	// Try to compress into a buffer of len - 1.
	// If that fails, save uncompressed.
	auto compressed = std::make_unique<char[]>(size - 1);
	std::uint32_t compressed_size = pcl::lzfCompress(data.get(), size, compressed.get(), size - 1);

	// Couldn't compress into the desired size,
	// so we're better off saving uncompressed.
	if (compressed_size == 0) {
		//TODO (@wko): Ignore for now, but needs fixing.
		return writePcdBinary(stream, cloud); //NOLINT
	}

	// Write PCD file.
	stream << pcl::PCDWriter::generateHeader(cloud) << "DATA binary_compressed\n";
	stream.write(reinterpret_cast<char const *>(&compressed_size), 4);
	stream.write(reinterpret_cast<char const *>(&size), 4);
	stream.write(compressed.get(), compressed_size);
}

template<typename PointT>
estd::result<void, estd::error> writePcdBinaryCompressed(std::string const & path, const pcl::PointCloud<PointT> &cloud) {
	// Calculate header and data size.
	std::string header = pcl::PCDWriter::generateHeader(cloud) + "DATA binary_compressed\n";

	// Transpose the cloud fields so all values for the same field are contiguous.
	auto [data, data_size] = impl::transposeCloudFields(cloud);
	std::size_t total_size = header.size() + 8 + data_size;

	// Create a new file of the right size and map it in memory.
	auto map = impl::MemoryMappedFileRw::create(path, total_size);
	if (!map) return map.error();

	char * dest = map->as<char>();

	// Try to compress. If it doesn't fit, write the raw cloud instead.
	std::uint32_t compressed_size = pcl::lzfCompress(data.get(), data_size, dest + header.size() + 8, data_size);
	if (!compressed_size) {
		//TODO (@wko) Need to fix.
		return writePcdBinary(path, cloud); //NOLINT
	}

	std::memcpy(dest, header.data(), header.size());
	std::memcpy(dest + header.size() + 0, &compressed_size, 4);
	std::memcpy(dest + header.size() + 4, &data_size,       4);

	map->truncate(header.size() + 8 + compressed_size);

	return estd::in_place_valid;
}

}
