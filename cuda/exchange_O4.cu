#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange.go for more details.
extern "C" __global__ void
addexchangeo4(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    uint8_t r0 = regions[I];
    float3 B  = make_float3(0.0,0.0,0.0);

    int i_;    // neighbor index
    float a__; // inter-cell exchange stiffness

    ////////////////////////////////////////////////////////////////////////////////////////
    // This is the FOURTH ORDER version.
    // - Assuming: ONLY ONE value for the exchange-parameter
    a__ = aLUT2d[symidx(r0, r0)];
    // - Assuming: There is a minimal number of lattice sites between holes

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
    //// FOURTH ORDER VERSION:
    //// mM2|mM1|m0|m1|m2
    

    ////////////////////////////////////////////////////////////////////////////////////////
    //// XXXXXXXXXXXXXXX

    if(PBCx){
        i_  = idx(lclampx(ix-2), iy, iz);
        float3 mM2 = make_float3(mx[i_], my[i_], mz[i_]);
        i_  = idx(lclampx(ix-1), iy, iz);
        float3 mM1  = make_float3(mx[i_], my[i_], mz[i_]);
        i_  = idx(hclampx(ix+1), iy, iz);
        float3 m1  = make_float3(mx[i_], my[i_], mz[i_]);
        i_  = idx(hclampx(ix+2), iy, iz);
        float3 m2 = make_float3(mx[i_], my[i_], mz[i_]);
        B += wx * a__ *((-1.f/12.f)*mM2 + (16.f/12.f)*mM1 - (30.f/12.f)*m0 + (16.f/12.f)*m1 - (1.f/12.f)*m2);
    }
    else{
        float3 mM2  = make_float3(0.0f, 0.0f, 0.0f);
        float3 mM1  = make_float3(0.0f, 0.0f, 0.0f);
        float3 m1  = make_float3(0.0f, 0.0f, 0.0f);
        float3 m2  = make_float3(0.0f, 0.0f, 0.0f);

        if( ix-2 > -1 ){
            i_  = idx(lclampx(ix-2), iy, iz);
            mM2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        if( ix-1 > -1 ) {
            i_  = idx(lclampx(ix-1), iy, iz);
            mM1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        if( ix+1 < Nx ) {
            i_  = idx(hclampx(ix+1), iy, iz);
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        if( ix+2 < Nx ) {
            i_  = idx(hclampx(ix+2), iy, iz);
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
            
        // CASE 1 --- some neighbor i<0 is outside sample:
        if(is0(mM2)){
            //// X X|m0 O O O . . . 
            if(is0(mM1)){
                i_  = idx(hclampx(ix+3), iy, iz);
                float3 m3 = make_float3(mx[i_], my[i_], mz[i_]);
                mM1 = (1.f/22.f) * (17.f*m0 + 9.f*m1 - 5.f*m2 + 1.f*m3);
                B += wx * a__ * ((11.f/12.f)*mM1 - (20.f/12.f)*m0 + (6.f/12.f)*m1 + (4.f/12.f)*m2 - (1.f/12.f)*m3);
            }
            //// X|O m0 O O . . . 
            else{
                mM2 = (1.f/22.f) * (17.f*mM1 + 9.f*m0 - 5.f*m1 + 1.f*m2);
                B += wx * a__ *(1.f/12.f)*(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2);
            }

        }
        // CASE 2 --- some neighbor i>0 is outside sample:
        else if(is0(m2)){
            //// O O O m0|X X  
            if(is0(m1)){
                i_  = idx(lclampx(ix-3), iy, iz);
                float3 mM3 = make_float3(mx[i_], my[i_], mz[i_]);
                m1 = (1.f/22.f) * (mM3 - 5.f*mM2 + 9.f*mM1 + 17.f*m0);
                B += wx * a__ * (1.f/12.f)*(-1.f*mM3 + 4.f*mM2 +6.f*mM1 - 20.f*m0 + 11.f*m1);
            }
            //// O O m0 O|X   
            else{
                m2 = (1.f/22.f) * (mM2 - 5.f*mM1 + 9.f*m0 + 17.f*m1);
                B += wx * a__ *(1.f/12.f)*(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2);
            }

        }
        else{
            B += wx * a__ *(1.f/12.f)*(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2);
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////
    //// YYYYYYYYYYYYYYYY

    if(PBCy){
        i_  = idx(ix, lclampy(iy-2), iz);
        float3 mM2 = make_float3(mx[i_], my[i_], mz[i_]);
        i_  = idx(ix, lclampy(iy-1), iz);
        float3 mM1  = make_float3(mx[i_], my[i_], mz[i_]);
        i_  = idx(ix, hclampy(iy+1), iz);
        float3 m1  = make_float3(mx[i_], my[i_], mz[i_]);
        i_  = idx(ix, hclampy(iy+2), iz);
        float3 m2 = make_float3(mx[i_], my[i_], mz[i_]);
        B += wy * a__ *(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2)*(1.f/12.f);
    }
    else{
        float3 mM2  = make_float3(0.0f, 0.0f, 0.0f);
        float3 mM1  = make_float3(0.0f, 0.0f, 0.0f);
        float3 m1  = make_float3(0.0f, 0.0f, 0.0f);
        float3 m2  = make_float3(0.0f, 0.0f, 0.0f);

        if( iy-2 > -1 ){
            i_  = idx(ix, lclampy(iy-2), iz);
            mM2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        if( iy-1 > -1 ) {
            i_  = idx(ix, lclampy(iy-1), iz);
            mM1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        if( iy+1 < Ny ) {
            i_  = idx(ix, hclampy(iy+1), iz);
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        if( iy+2 < Ny ) {
            i_  = idx(ix, hclampy(iy+2), iz);
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
            
        // CASE 1 --- some neighbor i<0 is outside sample:
        if(is0(mM2)){
            //// X X|m0 O O O . . . 
            if(is0(mM1)){
                i_  = idx(ix, hclampy(iy+3), iz);
                float3 m3 = make_float3(mx[i_], my[i_], mz[i_]);
                mM1 = (1.f/22.f) * (17.f*m0 + 9.f*m1 - 5.f*m2 + 1.f*m3);
                B += wy * a__ * (11.f*mM1 - 20.f*m0 + 6.f*m1 + 4.f*m2 - 1.f*m3)*(1.f/12.f);
            }
            //// X|O m0 O O . . . 
            else{
                mM2 = (1.f/22.f) * (17.f*mM1 + 9.f*m0 - 5.f*m1 + 1.f*m2);
                B += wy * a__ *(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2)*(1.f/12.f);
            }

        }
        // CASE 2 --- some neighbor i>0 is outside sample:
        else if(is0(m2)){
            //// O O O m0|X X  
            if(is0(m1)){
                i_  = idx(ix, lclampy(iy-3), iz);
                float3 mM3 = make_float3(mx[i_], my[i_], mz[i_]);
                m1 = (1.f/22.f) * (mM3 - 5.f*mM2 + 9.f*mM1 + 17.f*m0);
                B += wy * a__ * (-1.f*mM3 + 4.f*mM2 +6.f*mM1 - 20.f*m0 + 11.f*m1)*(1.f/12.f);
            }
            //// O O m0 O|X   
            else{
                m2 = (1.f/22.f) * (mM2 - 5.f*mM1 + 9.f*m0 + 17.f*m1);
                B += wy * a__ *(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2)*(1.f/12.f);
            }

        }
        else{
            B += wy * a__ *(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2)*(1.f/12.f);
        }
    }


    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        if(PBCz){
            i_  = idx(ix, iy, lclampz(iz-2));
            float3 mM2 = make_float3(mx[i_], my[i_], mz[i_]);
            i_  = idx(ix, iy, lclampz(iz-1));
            float3 mM1  = make_float3(mx[i_], my[i_], mz[i_]);
            i_  = idx(ix, iy, hclampz(iz+1));
            float3 m1  = make_float3(mx[i_], my[i_], mz[i_]);
            i_  = idx(ix, iy, hclampz(iz+2));
            float3 m2 = make_float3(mx[i_], my[i_], mz[i_]);
            B += wz * a__ *(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2)*(1.f/12.f);
        }
        else{
            float3 mM2  = make_float3(0.0f, 0.0f, 0.0f);
            float3 mM1  = make_float3(0.0f, 0.0f, 0.0f);
            float3 m1  = make_float3(0.0f, 0.0f, 0.0f);
            float3 m2  = make_float3(0.0f, 0.0f, 0.0f);

            if( iz-2 > -1 ){
                i_  = idx(ix, iy, lclampz(iz-2));
                mM2 = make_float3(mx[i_], my[i_], mz[i_]);
            }
            if( iz-1 > -1 ) {
                i_  = idx(ix, iy, lclampz(iz-1));
                mM1 = make_float3(mx[i_], my[i_], mz[i_]);
            }
            if( iz+1 < Nz ) {
                i_  = idx(ix, iy, hclampz(iz+1));
                m1 = make_float3(mx[i_], my[i_], mz[i_]);
            }
            if( iz+2 < Nz ) {
                i_  = idx(ix, iy, hclampz(iz+2));
                m2 = make_float3(mx[i_], my[i_], mz[i_]);
            }
                
            // CASE 1 --- some neighbor i<0 is outside sample:
            if(is0(mM2)){
                //// X X|m0 O O O . . . 
                if(is0(mM1)){
                    i_  = idx(ix, iy, hclampz(iz+3));
                    float3 m3 = make_float3(mx[i_], my[i_], mz[i_]);
                    mM1 = (1.f/22.f) * (17.f*m0 + 9.f*m1 - 5.f*m2 + 1.f*m3);
                    B += wz * a__ * (11.f*mM1 - 20.f*m0 + 6.f*m1 + 4.f*m2 - 1.f*m3)*(1.f/12.f);
                }
                //// X|O m0 O O . . . 
                else{
                    mM2 = (1.f/22.f) * (17.f*mM1 + 9.f*m0 - 5.f*m1 + 1.f*m2);
                    B += wz * a__ *(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2)*(1.f/12.f);
                }

            }
            // CASE 2 --- some neighbor i>0 is outside sample:
            else if(is0(m2)){
                //// O O O m0|X X  
                if(is0(m1)){
                    i_  = idx(ix, iy, lclampz(iz-3));
                    float3 mM3 = make_float3(mx[i_], my[i_], mz[i_]);
                    m1 = (1.f/22.f) * (mM3 - 5.f*mM2 + 9.f*mM1 + 17.f*m0);
                    B += wz * a__ * (-1.f*mM3 + 4.f*mM2 +6.f*mM1 - 20.f*m0 + 11.f*m1)*(1.f/12.f);
                }
                //// O O m0 O|X   
                else{
                    m2 = (1.f/22.f) * (mM2 - 5.f*mM1 + 9.f*m0 + 17.f*m1);
                    B += wz * a__ *(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2)*(1.f/12.f);
                }

            }
            else{
                B += wz * a__ *(-1.f*mM2 + 16.f*mM1 - 30.f*m0 + 16.f*m1 - 1.f*m2)*(1.f/12.f);
            }
        }
    }


    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Bx[I] += B.x*invMs;
    By[I] += B.y*invMs;
    Bz[I] += B.z*invMs;
}

