#!/bin/bash

# EluSight - Build Distribution Script
# This script creates the dist folder with .tar.gz and .whl files

echo "========================================="
echo "🔬 EluSight - Building Distribution"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Clean old dist folder
echo -e "\n${YELLOW}📁 Step 1: Cleaning old dist folder...${NC}"
rm -rf dist/
rm -rf build/
rm -rf *.egg-info
echo -e "${GREEN}✅ Cleaned old build files${NC}"

# Step 2: Check if build tool is installed
echo -e "\n${YELLOW}🔧 Step 2: Checking build tools...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found! Please install Python first.${NC}"
    exit 1
fi

# Install/upgrade build tools
python -m pip install --upgrade pip
python -m pip install build twine wheel
echo -e "${GREEN}✅ Build tools ready${NC}"

# Step 3: Build the package
echo -e "\n${YELLOW}📦 Step 3: Building package...${NC}"
python -m build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Package built successfully!${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi

# Step 4: Show created files
echo -e "\n${YELLOW}📁 Step 4: Generated files in dist/:${NC}"
ls -la dist/

# Step 5: Count and verify files
echo -e "\n${YELLOW}🔍 Step 5: Verifying files...${NC}"
TAR_COUNT=$(ls -1 dist/*.tar.gz 2>/dev/null | wc -l)
WHL_COUNT=$(ls -1 dist/*.whl 2>/dev/null | wc -l)

if [ $TAR_COUNT -gt 0 ] && [ $WHL_COUNT -gt 0 ]; then
    echo -e "${GREEN}✅ Found $TAR_COUNT .tar.gz and $WHL_COUNT .whl files${NC}"
    
    # Show file sizes
    echo -e "\n${GREEN}📊 File Details:${NC}"
    for file in dist/*; do
        SIZE=$(ls -lh "$file" | awk '{print $5}')
        echo "   📄 $(basename "$file") - $SIZE"
    done
else
    echo -e "${RED}❌ Missing files! Expected .tar.gz and .whl${NC}"
    exit 1
fi

# Step 6: Optional - Check package validity
echo -e "\n${YELLOW}🔍 Step 6: Checking package validity...${NC}"
python -m twine check dist/*
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Package passes PyPI checks!${NC}"
else
    echo -e "${YELLOW}⚠️  Package has warnings but can still be uploaded${NC}"
fi

# Final summary
echo -e "\n========================================="
echo -e "${GREEN}🎉 BUILD COMPLETE!${NC}"
echo -e "========================================="
echo -e "\n📁 dist folder created with:"
ls -1 dist/
echo -e "\n🚀 To upload to PyPI, run:"
echo -e "   ${YELLOW}python -m twine upload dist/*${NC}"
echo -e "\n🔬 To upload to Test PyPI first:"
echo -e "   ${YELLOW}python -m twine upload --repository testpypi dist/*${NC}"
echo -e "\n========================================="
