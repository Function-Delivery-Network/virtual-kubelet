#!/usr/bin/env bash
cp -r ../../../bin/ bin
sudo docker build --push --platform=linux/amd64 --tag isaacnez/virtual-kubelet:latest .
