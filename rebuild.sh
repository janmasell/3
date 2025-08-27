# make sure that in cuda/Makefile the gcc compiler is set correctly, matching your version of CUDA!
make realclean && make CUDA_CC=86

########################################
# for reference:

#####
# KIT-TFP Workstation GPUNC1:
# Tesla V100 SXM2 32GB
# CC = 70, 5120 cores, 32GB RAM, 4096bit, 898.0 GB/s, FP32 15.67 TFLOPS, FP64 7834 GFLOPS
# https://www.techpowerup.com/gpu-specs/tesla-v100-sxm2-32-gb.c3185

#####
# JM office:
# RTX 3070
# CC = 86, 5888 cores, 8GB RAM, 256bit, 448.0 GB/s, FP32 20.21 TFLOPS, FP64 317.4 GFLOPS
# https://www.techpowerup.com/gpu-specs/geforce-rtx-3070.c3674

#####
# JM RIKEN Workstation:
# RTX 3090
# CC = 86, 10496 cores, 24GB RAM, 384bit, 936.2 GB/s, FP32 35.58 TFLOPS, FP64 556.0 GFLOPS
# https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622

#####
# JM RIKEN Office:
# RTX 2070 super
# CC = 75, 2560 cores, 8GB RAM, 256bit, 448.0 GB/s, FP32 9.06 TFLOPS, FP64 283.2 GFLOPS
# https://www.techpowerup.com/gpu-specs/geforce-rtx-2070-super.c3440

#####
# JM RIKEN Office:
# TITAN V
# CC = 70, 5120 cores, 12GB RAM, 3072bit, 651.3 GB/s, FP32 14.9 TFLOPS, FP64 7450 GFLOPS
# https://www.techpowerup.com/gpu-specs/titan-v.c3051