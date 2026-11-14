#include "fusion/backend.hpp"
#include "fusion/image_io.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct TestFiles {
    fs::path directory;
    std::vector<fs::path> paths;

    TestFiles() {
        const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            directory = fs::temp_directory_path() /
                        ("fusion-image-io-" + std::to_string(seed) + "-" + std::to_string(attempt));
            if (fs::create_directory(directory)) return;
        }
        throw std::runtime_error("Cannot create image I/O test directory");
    }

    fs::path file(const char* name) {
        paths.push_back(directory / name);
        return paths.back();
    }

    ~TestFiles() {
        std::error_code error;
        for (const auto& path : paths) fs::remove(path, error);
        fs::remove(directory, error);
    }
};

void write_bytes(const fs::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("Cannot write test fixture");
}

bool matches(const fusion::RgbImage& image, std::size_t width, std::size_t height,
             const std::vector<unsigned char>& expected) {
    if (image.width != width || image.height != height || image.pixels.size() != expected.size()) {
        std::cerr << "Unexpected loaded image dimensions\n";
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(image.pixels[i] - expected[i] / 255.0f) > 1e-7f) {
            std::cerr << "Incorrect image value at element " << i << '\n';
            return false;
        }
    }
    return true;
}

template <typename Exception, typename Function>
bool rejects(const char* name, Function function) {
    try {
        function();
        std::cerr << name << " was accepted\n";
    } catch (const Exception&) {
        return true;
    } catch (const std::exception& error) {
        std::cerr << name << " produced an unexpected error: " << error.what() << '\n';
    }
    return false;
}

}  // namespace

int main(int argc, char* argv[]) {
    TestFiles files;
    bool ok = true;
    const std::vector<unsigned char> png{
        0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,0x49,0x48,0x44,0x52,
        0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x01,0x08,0x02,0x00,0x00,0x00,0x7b,0x40,0xe8,
        0xdd,0x00,0x00,0x00,0x04,0x67,0x41,0x4d,0x41,0x00,0x01,0x86,0xa0,0x31,0xe8,0x96,
        0x5f,0x00,0x00,0x00,0x0f,0x49,0x44,0x41,0x54,0x78,0x9c,0x63,0xe0,0x52,0xf8,0xff,
        0x9f,0xa1,0x1e,0x00,0x08,0x5b,0x02,0xa8,0xc8,0x63,0xcd,0x11,0x00,0x00,0x00,0x00,
        0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82,
    };
    const std::vector<unsigned char> rgba{
        0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,0x49,0x48,0x44,0x52,
        0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x08,0x06,0x00,0x00,0x00,0x1f,0x15,0xc4,
        0x89,0x00,0x00,0x00,0x04,0x67,0x41,0x4d,0x41,0x00,0x01,0x86,0xa0,0x31,0xe8,0x96,
        0x5f,0x00,0x00,0x00,0x0d,0x49,0x44,0x41,0x54,0x78,0x9c,0x63,0xe0,0x52,0xf8,0xff,
        0x1f,0x00,0x03,0x8a,0x02,0x29,0xab,0x0e,0x84,0xe1,0x00,0x00,0x00,0x00,0x49,0x45,
        0x4e,0x44,0xae,0x42,0x60,0x82,
    };
    const std::vector<unsigned char> rgb16{
        0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,0x49,0x48,0x44,0x52,
        0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x10,0x02,0x00,0x00,0x00,0xc0,0xe7,0x8f,
        0x9d,0x00,0x00,0x00,0x04,0x67,0x41,0x4d,0x41,0x00,0x01,0x86,0xa0,0x31,0xe8,0x96,
        0x5f,0x00,0x00,0x00,0x0f,0x49,0x44,0x41,0x54,0x78,0x9c,0x63,0x60,0xe0,0x62,0x50,
        0x60,0xf8,0x0f,0x00,0x01,0x98,0x01,0x2a,0xdc,0x21,0x9d,0x4e,0x00,0x00,0x00,0x00,
        0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82,
    };
    const auto png_path = files.file("fixture.png");
    write_bytes(png_path, png);
    const std::vector<unsigned char> original{10, 32, 255, 255, 0, 127};
    ok &= matches(fusion::load_image(png_path.string()), 2, 1, original);

    const auto rgba_path = files.file("rgba.png");
    const auto rgb16_path = files.file("rgb16.png");
    write_bytes(rgba_path, rgba);
    write_bytes(rgb16_path, rgb16);
    ok &= rejects<std::runtime_error>("PNG alpha", [&] { fusion::load_image(rgba_path.string()); });
    ok &= rejects<std::runtime_error>("16-bit PNG", [&] { fusion::load_image(rgb16_path.string()); });

    const auto ppm_path = files.file("fixture.ppm");
    const std::string header = "P6\n# comment\n2 1\n255\n";
    std::vector<unsigned char> ppm(header.begin(), header.end());
    ppm.insert(ppm.end(), original.begin(), original.end());
    write_bytes(ppm_path, ppm);
    ok &= matches(fusion::load_image(ppm_path.string()), 2, 1, original);
    const auto carriage_return = files.file("carriage-return.ppm");
    write_bytes(carriage_return, {'P','6',' ','1',' ','1',' ','2','5','5','\r', 10, 32, 255});
    ok &= matches(fusion::load_image(carriage_return.string()), 1, 1, {10, 32, 255});

    const fusion::RgbImage input{2, 1, {-1.0f, 0.5f, 2.0f, 10.0f / 255.0f, 32.0f / 255.0f, 1.0f}};
    const std::vector<unsigned char> encoded{0, 128, 255, 10, 32, 255};
    const auto output_png = files.file("output.PNG");
    const auto output_ppm = files.file("output.ppm");
    fusion::save_image(output_png.string(), input);
    fusion::save_image(output_ppm.string(), input);
    ok &= matches(fusion::load_image(output_png.string()), 2, 1, encoded);
    ok &= matches(fusion::load_image(output_ppm.string()), 2, 1, encoded);

    const auto corrupt = files.file("corrupt.png");
    write_bytes(corrupt, {0, 1, 2});
    ok &= rejects<std::runtime_error>("corrupt PNG", [&] { fusion::load_image(corrupt.string()); });
    const auto truncated = files.file("truncated.ppm");
    write_bytes(truncated, {'P','6','\n','1',' ','1','\n','2','5','5','\n', 1});
    ok &= rejects<std::runtime_error>("truncated PPM", [&] { fusion::load_image(truncated.string()); });
    const auto wrong_max = files.file("wrong-max.ppm");
    write_bytes(wrong_max, {'P','6','\n','1',' ','1','\n','1','5','\n', 1, 2, 3});
    ok &= rejects<std::runtime_error>("unsupported PPM max", [&] { fusion::load_image(wrong_max.string()); });
    ok &= rejects<std::invalid_argument>("unsupported extension", [&] { fusion::load_image("image.jpg"); });
    ok &= rejects<std::runtime_error>("missing input", [&] {
        fusion::load_image((files.directory / "missing.png").string());
    });
    ok &= rejects<std::runtime_error>("invalid output path", [&] {
        fusion::save_image((files.directory / "missing" / "output.png").string(), input);
    });
    ok &= rejects<std::invalid_argument>("bad dimensions", [&] {
        fusion::save_image(output_png.string(), {0, 1, {}});
    });
    ok &= rejects<std::invalid_argument>("bad pixel count", [&] {
        fusion::save_image(output_png.string(), {1, 1, {0.5f}});
    });
    ok &= rejects<std::invalid_argument>("non-finite output", [&] {
        fusion::save_image(output_png.string(), {1, 1, {0.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f}});
    });
    ok &= matches(fusion::load_image(output_png.string()), 2, 1, encoded);

    if (argc == 4) {
        std::vector<fusion::RgbImage> images;
        for (int i = 1; i < argc; ++i) images.push_back(fusion::load_image(argv[i]));
        auto backend = fusion::make_cpu_backend();
        const auto fused = backend->fuse(images, {});
        const auto result_path = files.file("fused.png");
        fusion::save_image(result_path.string(), fused);
        const auto loaded = fusion::load_image(result_path.string());
        if (loaded.width != images[0].width || loaded.height != images[0].height) {
            std::cerr << "Real input integration dimensions do not match\n";
            ok = false;
        }
    } else if (argc != 1) {
        std::cerr << "Expected zero or three input paths\n";
        return 1;
    }
    return ok ? 0 : 1;
}
