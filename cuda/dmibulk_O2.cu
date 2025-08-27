#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

//##################
//
// Exchange + Dzyaloshinskii-Moriya interaction for bulk material.
// E = A (grad m)^2 + D m.rot m
//
// Effective field:
// B = -(1/Ms) dE/dm 
//   = 2 Aex/Ms Laplace m - 2 D/Ms rot m
//     | 2 Aex/Ms (d/dx2+d/dy2+d/dz2)mx - 2 D/Ms (dmzdy - dmydz) |
//   = | 2 Aex/Ms (d/dx2+d/dy2+d/dz2)my - 2 D/Ms (dmxdz - dmzdx) |
//     | 2 Aex/Ms (d/dx2+d/dy2+d/dz2)mz - 2 D/Ms (dmydx - dmxdy) |
//
// von Neumann boundary conditions:
// dM/dn = (D/2A) n x M   for a given edge perp to n
//
//##################

extern "C" __global__ void
adddmibulko2(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
           float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
           float* __restrict__ Ms_, float Ms_mul,
           float* __restrict__ aLUT2d, float* __restrict__ DLUT2d,
           uint8_t* __restrict__ regions,
           float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    float3 h = make_float3(0.0,0.0,0.0);          // add to H
    float3 m0 = make_float3(mx[I], my[I], mz[I]); // central m
    uint8_t r0 = regions[I];
    int i_;                                       // neighbor index

    if(is0(m0)) {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    // This is the FOURTH ORDER version.
    // - Assuming: ONLY ONE value for the exchange-parameter
    float A = aLUT2d[symidx(r0, r0)];
    float D = DLUT2d[symidx(r0, r0)];
    const float A_cx2 = A/(cx*cx);
    const float D_cx  = D/cx;
    const float A_cy2 = A/(cy*cy);
    const float D_cy  = D/cy;
    const float A_cz2 = A/(cz*cz);
    const float D_cz  = D/cz;
    // - Assuming: There is a minimal number of lattice sites between holes


    

    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // x derivatives
    if( Nx==1 ){
        // do nothing.
    }
    else if ( (Nx >= 2) || PBCx) {

        // neighbor at x-1
        float3 mm1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(lclampx(ix-1), iy, iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( ix-1 >= 0 || PBCx ){ mm1 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+1
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(hclampx(ix+1), iy, iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( ix+1 < Nx || PBCx ){ m1  = make_float3(mx[i_], my[i_], mz[i_]); }

        // anyway only one spin with holes around?
        if( is0(mm1) && is0(m1) ){
            // do nothing.
        }

        // no holes/edges, all fine:
        // 0 m 0
        else if( !is0(mm1) && !is0(m1) ){
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * ((m1+mm1) - 2.f*m0);
            // dmi = - 2 D/Ms rot m
            h.x -= 0.f; // nothing happening here
            h.y -= (2.0f*D_cx ) * (-1.f) * (0.5f*(m1.z-mm1.z));
            h.z -= (2.0f*D_cx ) *          (0.5f*(m1.y-mm1.y));
        }

        // X|m 0 
        else if( is0(mm1) && !is0(m1) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(0.03515625f + A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.125f*m0.x - 0.125f*m1.x;
            me.y = pref*( 0.210938f*m0.z - 0.0234375f*m1.z + A_aD*( 1.125f*m0.y - 0.125f*m1.y ) );
            me.z = pref*(-0.210938f*m0.y + 0.0234375f*m1.y + A_aD*( 1.125f*m0.z - 0.125f*m1.z ) );
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (2.66667f*me - 4.f*m0 + 1.33333f*m1);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * (-1.33333f*me.z + 1.f*m0.z + 0.333333f*m1.z );
            h.z -= (2.0f*D_cx ) *          (-1.33333f*me.y + 1.f*m0.y + 0.333333f*m1.y );
        }

        // 0 m|X 
        else if( !is0(mm1) && is0(m1) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(0.03515625f + A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x =-1.125f*m0.x + 0.125f*mm1.x;
            me.y = pref*(-0.210938f*m0.z + 0.0234375f*mm1.z + A_aD*( 1.125f*m0.y - 0.125f*mm1.y ) );
            me.z = pref*( 0.210938f*m0.y - 0.0234375f*mm1.y + A_aD*( 1.125f*m0.z - 0.125f*mm1.z ) );
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (2.66667f*me - 4.f*m0 + 1.33333f*mm1);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * ( 1.33333f*me.z - 1.f*m0.z - 0.333333f*mm1.z );
            h.z -= (2.0f*D_cx ) *          ( 1.33333f*me.y - 1.f*m0.y - 0.333333f*mm1.y );
        }
        // other
        else{
            printf("ERROR: dmibulk_O2.cu: Undefined case along x."); return;
        }
    }

    
        // SHORTCUT FOR THE CREATION OF THESE LINES USING THE X-CASE: 
        // Take lines from x-derivatives and permute (x,y,z)->(y,z,x), 
        // i.e., by replacing .x->.i, .z->.x, .y->.z, .i->.y
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // y derivatives
    if( Ny==1 ){
        // do nothing.
    }
    else if ( (Ny >= 2) || PBCy) {

        // neighbor at y-1
        float3 mm1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( iy-1 >= 0 || PBCy ){ mm1 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at y+1
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( iy+1 < Ny || PBCy ){ m1  = make_float3(mx[i_], my[i_], mz[i_]); }

        // anyway only one spin with holes around?
        if( is0(mm1) && is0(m1) ){
            // do nothing.
        }

        // no holes/edges, all fine:
        // 0 m 0
        else if( !is0(mm1) && !is0(m1) ){
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * ((m1+mm1) - 2.f*m0);
            // dmi = - 2 D/Ms rot m
            h.y -= 0.f; // nothing happening here
            h.z -= (2.0f*D_cy ) * (-1.f) * (0.5f*(m1.x-mm1.x));
            h.x -= (2.0f*D_cy ) *          (0.5f*(m1.z-mm1.z));
        }

        // y|m 0 
        else if( is0(mm1) && !is0(m1) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(0.03515625f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallz at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.125f*m0.y - 0.125f*m1.y;
            me.z = pref*( 0.210938f*m0.x - 0.0234375f*m1.x + A_aD*( 1.125f*m0.z - 0.125f*m1.z ) );
            me.x = pref*(-0.210938f*m0.z + 0.0234375f*m1.z + A_aD*( 1.125f*m0.x - 0.125f*m1.x ) );
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (2.66667f*me - 4.f*m0 + 1.33333f*m1);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * (-1.33333f*me.x + 1.f*m0.x + 0.333333f*m1.x );
            h.x -= (2.0f*D_cy ) *          (-1.33333f*me.z + 1.f*m0.z + 0.333333f*m1.z );
        }

        // 0 m|y 
        else if( !is0(mm1) && is0(m1) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(0.03515625f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallz at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y =-1.125f*m0.y + 0.125f*mm1.y;
            me.z = pref*(-0.210938f*m0.x + 0.0234375f*mm1.x + A_aD*( 1.125f*m0.z - 0.125f*mm1.z ) );
            me.x = pref*( 0.210938f*m0.z - 0.0234375f*mm1.z + A_aD*( 1.125f*m0.x - 0.125f*mm1.x ) );
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (2.66667f*me - 4.f*m0 + 1.33333f*mm1);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * ( 1.33333f*me.x - 1.f*m0.x - 0.333333f*mm1.x );
            h.x -= (2.0f*D_cy ) *          ( 1.33333f*me.z - 1.f*m0.z - 0.333333f*mm1.z );
        }
        // other
        else{
            printf("ERROR: dmibulk_O2.cu: Undefined case along y."); return;
        }
    }

    
        // SHORTCUT FOR THE CREATION OF THESE LINES USING THE Y-CASE: 
        // Take lines from y-derivatives and permute (x,y,z)->(y,z,x), 
        // i.e., by replacing .x->.i, .z->.x, .y->.z, .i->.y
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // z derivatives
    if( Nz==1 ){
        // do nothing.
    }
    else if ( (Nz >= 2) || PBCz) {

        // neighbor at z-1
        float3 mm1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-1));            // load neighbor m if inside grid, keep 0 otherwise
        if( iz-1 >= 0 || PBCz ){ mm1 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at z+1
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+1));            // load neighbor m if inside grid, keep 0 otherwise
        if( iz+1 < Nz || PBCz ){ m1  = make_float3(mx[i_], my[i_], mz[i_]); }

        // anyway only one spin with holes around?
        if( is0(mm1) && is0(m1) ){
            // do nothing.
        }

        // no holes/edges, all fine:
        // 0 m 0
        else if( !is0(mm1) && !is0(m1) ){
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * ((m1+mm1) - 2.f*m0);
            // dmi = - 2 D/Ms rot m
            h.z -= 0.f; // nothing happening here
            h.x -= (2.0f*D_cz ) * (-1.f) * (0.5f*(m1.y-mm1.y));
            h.y -= (2.0f*D_cz ) *          (0.5f*(m1.x-mm1.x));
        }

        // z|m 0 
        else if( is0(mm1) && !is0(m1) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(0.03515625f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallx at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.125f*m0.z - 0.125f*m1.z;
            me.x = pref*( 0.210938f*m0.y - 0.0234375f*m1.y + A_aD*( 1.125f*m0.x - 0.125f*m1.x ) );
            me.y = pref*(-0.210938f*m0.x + 0.0234375f*m1.x + A_aD*( 1.125f*m0.y - 0.125f*m1.y ) );
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (2.66667f*me - 4.f*m0 + 1.33333f*m1);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * (-1.33333f*me.y + 1.f*m0.y + 0.333333f*m1.y );
            h.y -= (2.0f*D_cz ) *          (-1.33333f*me.x + 1.f*m0.x + 0.333333f*m1.x );
        }

        // 0 m|z 
        else if( !is0(mm1) && is0(m1) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(0.03515625f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallx at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z =-1.125f*m0.z + 0.125f*mm1.z;
            me.x = pref*(-0.210938f*m0.y + 0.0234375f*mm1.y + A_aD*( 1.125f*m0.x - 0.125f*mm1.x ) );
            me.y = pref*( 0.210938f*m0.x - 0.0234375f*mm1.x + A_aD*( 1.125f*m0.y - 0.125f*mm1.y ) );
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (2.66667f*me - 4.f*m0 + 1.33333f*mm1);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * ( 1.33333f*me.y - 1.f*m0.y - 0.333333f*mm1.y );
            h.y -= (2.0f*D_cz ) *          ( 1.33333f*me.x - 1.f*m0.x - 0.333333f*mm1.x );
        }
        // other
        else{
            printf("ERROR: dmibulk_O2.cu: Undefined case along z."); return;
        }
    }











    

    // write back, result is H + Hdmi + Hex
    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Hx[I] += h.x*invMs;
    Hy[I] += h.y*invMs;
    Hz[I] += h.z*invMs;
}
