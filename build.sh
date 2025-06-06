#!/bin/bash
# build.sh - Simplified build script for PPOkemon project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Debug"
CLEAN=false
RUN_TESTS=false
JOBS=$(nproc)

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --release     Build in Release mode (default: Debug)"
    echo "  -c, --clean       Clean build directory before building"
    echo "  -t, --test        Run tests after building"
    echo "  -j, --jobs N      Number of parallel jobs (default: $(nproc))"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Debug build"
    echo "  $0 -r             # Release build"
    echo "  $0 -c -r -t       # Clean release build with tests"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== PPOkemon Build Script ===${NC}"
echo -e "${YELLOW}Build Type: ${BUILD_TYPE}${NC}"
echo -e "${YELLOW}Jobs: ${JOBS}${NC}"

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..

# Build
echo -e "${YELLOW}Building with ${JOBS} parallel jobs...${NC}"
make -j${JOBS}

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
    
    # List built artifacts
    echo -e "${BLUE}Built artifacts:${NC}"
    ls -lh ppokemon libppokemon_libs.a 2>/dev/null || true
    ls -lh tests/test_* 2>/dev/null || true
    
    # Run tests if requested
    if [ "$RUN_TESTS" = true ]; then
        echo -e "${YELLOW}Running tests...${NC}"
        make test
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ All tests passed!${NC}"
        else
            echo -e "${RED}❌ Some tests failed!${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}🎉 Build process completed successfully!${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
