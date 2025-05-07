#!/bin/bash

set -e

echo "NVIDIA NGC Login Helper"
echo "------------------------"
echo "1. Open https://ngc.nvidia.com/signin in your browser."
echo "2. Log in or create an account."
echo "3. Once logged in, go to your profile -> 'Setup' -> 'Get API Key'."
echo "4. Generate and copy your API key."

read -p $'\nPaste your NVIDIA NGC API key here (input will be hidden): ' -s NGC_API_KEY
echo ""

if [ -z "$NGC_API_KEY" ]; then
  echo "No API key provided. Exiting."
  exit 1
fi

echo "Logging in to nvcr.io..."

echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

if [ $? -eq 0 ]; then
  echo "Successfully logged in to nvcr.io."
else
  echo "Login failed. Please check your API key and try again."
  exit 1
fi

