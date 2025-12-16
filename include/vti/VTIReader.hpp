//needsoverhaul
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cmath>

struct VTIResult {
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;
    std::vector<std::uint8_t> mask;
};

/**
 * Reads a .vti file and creates a binary mask.
 *
 * Assumptions:
 *  - File is VTK ImageData (.vti) in XML format
 *  - WholeExtent or Extent attribute is present
 *  - First DataArray contains ASCII data
 *  - DataArray contains either:
 *      * Nx * Ny * Nz scalar values, or
 *      * 3 * Nx * Ny * Nz vector values
 *  - For vector data, magnitude sqrt(x^2 + y^2 + z^2) is used
 *  - Values > threshold are mapped to mask = 1, else 0
 */

inline VTIResult readVTI(const std::string& filename, double threshold = 0.5)
{
    //read file
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("[VTIReader]: Fatal, could not open File: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    VTIResult result;

    //Parse Extent, WholeExtent or Extent

    auto findAttribute = [&](const std::string& name) -> std::string {
        std::string pattern = name + "=\"";
        auto pos = content.find(pattern);
        if (pos == std::string::npos) return {};
        pos += pattern.size();
        auto end = content.find('"', pos);
        if (end == std::string::npos) return {};
        return content.substr(pos, end - pos);
    };

    std::string extentStr = findAttribute("WholeExtent");
    if (extentStr.empty()) {
        extentStr = findAttribute("Extent");
    }
    if (extentStr.empty()) {
        throw std::runtime_error("[VTIReader]: Fatal, no Extent/WholeExtent found in VTI File!");
    }

    // Format: x0 x1 y0 y1 z0 z1
    int x0, x1, y0, y1, z0, z1;
    {
        std::istringstream iss(extentStr);
        if (!(iss >> x0 >> x1 >> y0 >> y1 >> z0 >> z1)) {
            throw std::runtime_error("[VTIReader]: Fatal, could not parse Extent/WholeExtent! - " + extentStr);
        }
    }

    result.Nx = x1 - x0 + 1;
    result.Ny = y1 - y0 + 1;
    result.Nz = z1 - z0 + 1;

    const std::size_t expectedSize = static_cast<std::size_t>(result.Nx) *
                                     static_cast<std::size_t>(result.Ny) *
                                     static_cast<std::size_t>(result.Nz);

    //Find DataArray, first Scalar or Vector Data

    const std::string dataArrayTag = "<DataArray";
    auto daStart = content.find(dataArrayTag);
    if (daStart == std::string::npos) {
        throw std::runtime_error("[VTIReader]: Fatal, no <DataArray> found in VTI File.");
    }

    //end of tag
    auto gtPos = content.find('>', daStart);
    if (gtPos == std::string::npos) {
        throw std::runtime_error("[VTIReader]: Fatal, invalid DataArray-Tag.");
    }

    //end of DataArray tag
    auto endTag = content.find("</DataArray>", gtPos);
    if (endTag == std::string::npos) {
        throw std::runtime_error("[VTIReader]: Fatal, could not found closing </DataArray>-Tag!");
    }

    //Raw Data betweeen
    std::string dataStr = content.substr(gtPos + 1, endTag - (gtPos + 1));

    //Parse Values

    std::istringstream dataStream(dataStr);
    std::vector<double> values;
    values.reserve(expectedSize);

    double val;
    while (dataStream >> val) {
        values.push_back(val);
    }

    if (values.empty()) {
        throw std::runtime_error("[VTIReader]: Fatal, DataArray does not contain numerical values!");
    }

    std::size_t nVals = values.size();

    // Case 1: Scalar Values
    if (nVals == expectedSize) {
        result.mask.assign(expectedSize, 0);
        for (std::size_t i = 0; i < expectedSize; ++i) {
            result.mask[i] = (values[i] > threshold) ? 1u : 0u;
        }
    }
    // Case 2: Vector Values (RGB for example)
    else if (nVals == expectedSize * 3) {
        std::cerr << "[VTIReader]: The DataArray seems to contain 3 Components per Cell - RGB?\n";

        result.mask.assign(expectedSize, 0);
        for (std::size_t i = 0; i < expectedSize; ++i) {
            double vx = values[3 * i + 0];
            double vy = values[3 * i + 1];
            double vz = values[3 * i + 2];
            double mag = std::sqrt(vx * vx + vy * vy + vz * vz);
            result.mask[i] = (mag > threshold) ? 1u : 0u;
        }
    }
    // Case 3: ?? Garbage?
    else {
        std::cerr << "[VTIReader]: Warning: Number of values (" << nVals
                  << ") does not fit to Nx*Ny*Nz (" << expectedSize
                  << ") or 3*Nx*Ny*Nz. Mask will be cut and there might be problems...\n";

        std::size_t n = std::min(nVals, expectedSize);
        result.mask.assign(expectedSize, 0);
        for (std::size_t i = 0; i < n; ++i) {
            result.mask[i] = (values[i] > threshold) ? 1u : 0u;
        }
    }

    return result;
}
