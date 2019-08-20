#include "write.hpp"

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace dr::impl {
	std::uint32_t totalFieldSize(std::vector<pcl::PCLPointField> const & fields) {
		std::size_t total_size = 0;
		for (pcl::PCLPointField const & field : fields) {
			total_size += field.count * pcl::getFieldSize(field.datatype);
		}
		return total_size;
	}

	std::vector<std::size_t> calculateTransposedFieldOffsets(std::vector<pcl::PCLPointField> const & fields, std::size_t points) {
		std::vector<std::size_t> offsets;
		offsets.reserve(fields.size());

		offsets.push_back(0);
		for (std::size_t i = 1; i < fields.size(); ++i) {
			offsets.push_back(offsets.back() + pcl::getFieldSize(fields[i-1].datatype) * points);
		}

		return offsets;
	}

	estd::result<std::ofstream, estd::error> openFileForWriting(std::string const & path) {
		std::ofstream stream(path);
		if (!stream.good()) { return estd::error(std::error_code(errno, std::system_category()), "error opening file for writing: " + path); }
		return stream;
	}

	namespace {
		estd::error posixError(int error, std::string message) {
			return estd::error{std::error_code{error, std::system_category()}, std::move(message)};
		}

		estd::error lastError(std::string message) {
			return posixError(errno, std::move(message));
		}

		int64_t page_size = sysconf(_SC_PAGE_SIZE);

		std::size_t next_page_boundary(std::size_t address) {
			return (address + page_size - 1) / page_size * page_size;
		}
	} //namespace

	MemoryMappedFileRw::~MemoryMappedFileRw() {
		if (fd_ > -1) { ::close(fd_); }
		if (!static_cast<bool>(data_)) { return; }
		// Can't really handle errors here, so we'll just leak I suppose..
		// It's really unlikely that this fails though.
		::munmap(data_, length_);
		data_ = nullptr;
	}

	MemoryMappedFileRw::MemoryMappedFileRw(MemoryMappedFileRw && other) noexcept : fd_{other.fd_}, data_{other.data_}, length_{other.length_} {
		other.fd_   = -1;
		other.data_ = nullptr;
	}

	MemoryMappedFileRw & MemoryMappedFileRw::operator=(MemoryMappedFileRw && other) noexcept {
		if (fd_ > -1) { ::close(fd_); }
		if (static_cast<bool>(data_)) { ::munmap(data_, length_); }
		fd_     = std::exchange(other.fd_, -1);
		data_   = std::exchange(other.data_, nullptr);
		length_ = other.length_;
		return *this;
	}

	estd::result<MemoryMappedFileRw, estd::error> MemoryMappedFileRw::create(std::string const & path, std::size_t size) {
		constexpr int PERMISSION_FLAGS = 0644;
		int fd = ::openat(AT_FDCWD, path.c_str(), O_RDWR | O_CLOEXEC | O_CREAT | O_TRUNC, PERMISSION_FLAGS);
		if (fd == -1) { return lastError("failed to open file for writing: " + path); }

		// Make sure disk space is available to prevent bus errors.
		if (int error = ::posix_fallocate(fd, 0, size)) { return posixError(error, "failed to allocate disk space"); }

		return from_fd(fd, size);
	}

	estd::result<MemoryMappedFileRw, estd::error> MemoryMappedFileRw::from_fd(int fd, std::size_t size) {
		void * map = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (map == MAP_FAILED) { return lastError("failed to map FD " + std::to_string(fd)); } //NOLINT
		return MemoryMappedFileRw{fd, map, size};
	}

	estd::result<void, estd::error> MemoryMappedFileRw::truncate(std::size_t new_size) {
		if (new_size > this->length_) { return estd::error{std::errc::invalid_argument, "requested truncated size is larger than current file size"}; }
		if (::ftruncate(fd_, new_size) != 0) { return lastError("failed to truncate file"); }

		std::size_t drop_start  = next_page_boundary(std::size_t(this->data_) + new_size);
		std::size_t drop_length = std::size_t(this->data_) + this->length_ - drop_start;
		::munmap(reinterpret_cast<void *>(drop_start), drop_length);
		this->length_ = new_size;

		return estd::in_place_valid;
	}
} //namespace dr::impl

