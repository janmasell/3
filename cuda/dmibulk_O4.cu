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
adddmibulko4(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
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
    // note: proper O(a^4) needs at least 4 sites to work
    if( Nx==1 ){
        // do nothing.
    }
    else if ( (Nx >= 4) || PBCx) {

        // neighbor at x-2
        float3 mm2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(lclampx(ix-2), iy, iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( ix-2 >= 0 || PBCx ){ mm2 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x-1
        float3 mm1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(lclampx(ix-1), iy, iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( ix-1 >= 0 || PBCx ){ mm1 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+1
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(hclampx(ix+1), iy, iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( ix+1 < Nx || PBCx ){ m1  = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+2
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(hclampx(ix+2), iy, iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( ix+2 < Nx || PBCx ){ m2  = make_float3(mx[i_], my[i_], mz[i_]); }

        // check if holes are above and below
        if( (is0(mm2)||is0(mm1)) && (is0(m2)||is0(m1)) ){
            printf("ERROR: bulkO4.cu: Holes along x are too close by, needs at least 4 pixels of material."); return;
        }

        // now go case by case:

        // no holes/edges, all fine:
        // 0 0 m 0 0
        if( !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) ){
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (-0.0833333f*(mm2+m2) + 1.33333f*(mm1+m1) - 2.50000f*m0);
            // dmi = - 2 D/Ms rot m
            h.x -= 0.f; // nothing happening here
            h.y -= (2.0f*D_cx ) * (-1.f) * (0.0833333f*(mm2.z-m2.z) - 0.666667f*(mm1.z-m1.z));
            h.z -= (2.0f*D_cx ) *          (0.0833333f*(mm2.y-m2.y) - 0.666667f*(mm1.y-m1.y));
        }

        // 1 hole on mm2:
        // X|0 m 0 0
        else if( is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(44100.f + 1982464.f*A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.30504f*mm1.x - 0.435014f*m0.x + 0.156605f*m1.x - 0.0266335f*m2.x;
            me.y = pref*( 385875.f*mm1.z - 128625.f*m0.z + 46305.f*m1.z - 7875.f*m2.z + A_aD*( 2587200.f*mm1.y - 862400.f*m0.y + 310464.f*m1.y - 52800.f*m2.y ) );
            me.z = pref*(-385875.f*mm1.y + 128625.f*m0.y - 46305.f*m1.y + 7875.f*m2.y + A_aD*( 2587200.f*mm1.z - 862400.f*m0.z + 310464.f*m1.z - 52800.f*m2.z ) );
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (-0.304762f*me + 1.66667f*mm1 - 2.66667f*m0 + 1.4f*m1 - 0.0952381f*m2);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * (0.304762f*me.z - 1.f*mm1.z + 0.166667f*m0.z + 0.6f*m1.z - 0.0714286f*m2.z );
            h.z -= (2.0f*D_cx ) *          (0.304762f*me.y - 1.f*mm1.y + 0.166667f*m0.y + 0.6f*m1.y - 0.0714286f*m2.y );
        }

        // holes on mm2 & mm1: (not actually O(a^4) if we don take extra spins)
        // X X|m 0 0 
        else if( is0(mm2) && is0(mm1) && !is0(m1) && !is0(m2) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(450.f + 16928.f*A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.22283f*m0.x - 0.271739f*m1.x + 0.048913f*m2.x;
            me.y = pref*( 3375.f*m0.z - 750.f*m1.z + 135.f*m2.z + A_aD*( 20700.f*m0.y - 4600.f*m1.y + 828.f*m2.y ) );
            me.z = pref*(-3375.f*m0.y + 750.f*m1.y - 135.f*m2.y + A_aD*( 20700.f*m0.z - 4600.f*m1.z + 828.f*m2.z ) );
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (3.2f*me - 5.f*m0 + 2.f*m1 - 0.2f*m2);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * ( - 1.06667f*me.z + 0.5f*m0.z + 0.666667f*m1.z - 0.1f*m2.z );
            h.z -= (2.0f*D_cx ) *          ( - 1.06667f*me.y + 0.5f*m0.y + 0.666667f*m1.y - 0.1f*m2.y );
        }

        // holes on mm2 & mm1: (not actually O(a^4) if we don take extra spins)
        // 0 0 m|X X 
        else if( !is0(mm2) && !is0(mm1) && is0(m1) && is0(m2) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(450.f + 16928.f*A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.22283f*m0.x - 0.271739f*mm1.x + 0.048913f*mm2.x;
            me.y = pref*(-3375.f*m0.z + 750.f*mm1.z - 135.f*mm2.z + A_aD*( 20700.f*m0.y - 4600.f*mm1.y + 828.f*mm2.y ) );
            me.z = pref*( 3375.f*m0.y - 750.f*mm1.y + 135.f*mm2.y + A_aD*( 20700.f*m0.z - 4600.f*mm1.z + 828.f*mm2.z ) );
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (3.2f*me - 5.f*m0 + 2.f*mm1 - 0.2f*mm2);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * (   1.06667f*me.z - 0.5f*m0.z - 0.666667f*mm1.z + 0.1f*mm2.z );
            h.z -= (2.0f*D_cx ) *          (   1.06667f*me.y - 0.5f*m0.y - 0.666667f*mm1.y + 0.1f*mm2.y );
        }
        // 1 hole on m2:
        // 0 0 m 0|X
        else if( !is0(mm2) && !is0(mm1) && !is0(m1) && is0(m2) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(44100.f + 1982464.f*A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.30504f*m1.x - 0.435014f*m0.x + 0.156605f*mm1.x - 0.0266335f*mm2.x;
            me.y = pref*(-385875.f*m1.z + 128625.f*m0.z - 46305.f*mm1.z + 7875.f*mm2.z + A_aD*( 2587200.f*m1.y - 862400.f*m0.y + 310464.f*mm1.y - 52800.f*mm2.y ) );
            me.z = pref*( 385875.f*m1.y - 128625.f*m0.y + 46305.f*mm1.y - 7875.f*mm2.y + A_aD*( 2587200.f*m1.z - 862400.f*m0.z + 310464.f*mm1.z - 52800.f*mm2.z ) );
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (-0.304762f*me + 1.66667f*m1 - 2.66667f*m0 + 1.4f*mm1 - 0.0952381f*mm2);
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * (-0.304762f*me.z + 1.f*m1.z - 0.166667f*m0.z - 0.6f*mm1.z + 0.0714286f*mm2.z );
            h.z -= (2.0f*D_cx ) *          (-0.304762f*me.y + 1.f*m1.y - 0.166667f*m0.y - 0.6f*mm1.y + 0.0714286f*mm2.y );
        }
        // other
        else{
            printf("ERROR: bulkO4.cu: Undefined case along x."); return;
        }
    }
    else{
        printf("ERROR: bulkO4.cu: O(a^4) requires at least 4 pixels or PBC along x."); return;
    }



    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // y derivatives 
    // note: proper O(a^4) needs at least 4 sites to work
    if( Ny==1 ){
        // do nothing.
    }
    else if ( (Ny >= 4) || PBCy) {

        // neighbor at y-2
        float3 mm2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-2), iz);
        if( iy-2 >= 0 || PBCy ){ mm2 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at y-1
        float3 mm1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if( iy-1 >= 0 || PBCy ){ mm1 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at y+1
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if( iy+1 < Ny || PBCy ){ m1  = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at y+2
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+2), iz);
        if( iy+2 < Ny || PBCy ){ m2  = make_float3(mx[i_], my[i_], mz[i_]); }

        // check if holes are above and below
        if( (is0(mm2)||is0(mm1)) && (is0(m2)||is0(m1)) ){
            printf("ERROR: bulkO4.cu: Holes along y are too close by, needs at least 4 pixels of material."); return;
        }

        // now go case by case:

        // SHORTCUT: 
        // Take lines from x-derivatives and permute (x,y,z)->(y,z,x), 
        // i.e., by replacing .x->.i, .z->.x, .y->.z, .i->.y

        // no holes/edges, all fine:
        // 0 0 m 0 0
        if( !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) ){
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cy2) * (-0.0833333f*(mm2+m2) + 1.33333f*(mm1+m1) - 2.50000f*m0);
            // dmi = - 2 D/Ms rot m
            h.x -= (2.0f*D_cy ) *          (0.0833333f*(mm2.z-m2.z) - 0.666667f*(mm1.z-m1.z));
            h.y -= 0.f; // nothing happening here
            h.z -= (2.0f*D_cy ) * (-1.f) * (0.0833333f*(mm2.x-m2.x) - 0.666667f*(mm1.x-m1.x));
        }

        // 1 hole on mm2:
        // X|0 m 0 0
        else if( is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(44100.f + 1982464.f*A_aD*A_aD);
            // make me the vyrtual spyn on the edge (actuallz at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.30504f*mm1.y - 0.435014f*m0.y + 0.156605f*m1.y - 0.0266335f*m2.y;
            me.z = pref*( 385875.f*mm1.x - 128625.f*m0.x + 46305.f*m1.x - 7875.f*m2.x + A_aD*( 2587200.f*mm1.z - 862400.f*m0.z + 310464.f*m1.z - 52800.f*m2.z ) );
            me.x = pref*(-385875.f*mm1.z + 128625.f*m0.z - 46305.f*m1.z + 7875.f*m2.z + A_aD*( 2587200.f*mm1.x - 862400.f*m0.x + 310464.f*m1.x - 52800.f*m2.x ) );
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (-0.304762f*me + 1.66667f*mm1 - 2.66667f*m0 + 1.4f*m1 - 0.0952381f*m2);
            // dmy = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * (0.304762f*me.x - 1.f*mm1.x + 0.166667f*m0.x + 0.6f*m1.x - 0.0714286f*m2.x );
            h.x -= (2.0f*D_cy ) *          (0.304762f*me.z - 1.f*mm1.z + 0.166667f*m0.z + 0.6f*m1.z - 0.0714286f*m2.z );
        }

        // holes on mm2 & mm1: (not actuallz O(a^4) if we don take eitra spins)
        // i i|m 0 0 
        else if( is0(mm2) && is0(mm1) && !is0(m1) && !is0(m2) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(450.f + 16928.f*A_aD*A_aD);
            // make me the vyrtual spyn on the edge (actuallz at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.22283f*m0.y - 0.271739f*m1.y + 0.048913f*m2.y;
            me.z = pref*( 3375.f*m0.x - 750.f*m1.x + 135.f*m2.x + A_aD*( 20700.f*m0.z - 4600.f*m1.z + 828.f*m2.z ) );
            me.x = pref*(-3375.f*m0.z + 750.f*m1.z - 135.f*m2.z + A_aD*( 20700.f*m0.x - 4600.f*m1.x + 828.f*m2.x ) );
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (3.2f*me - 5.f*m0 + 2.f*m1 - 0.2f*m2);
            // dmy = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * ( - 1.06667f*me.x + 0.5f*m0.x + 0.666667f*m1.x - 0.1f*m2.x );
            h.x -= (2.0f*D_cy ) *          ( - 1.06667f*me.z + 0.5f*m0.z + 0.666667f*m1.z - 0.1f*m2.z );
        }

        // holes on mm2 & mm1: (not actuallz O(a^4) if we don take eitra spins)
        // 0 0 m|i i 
        else if( !is0(mm2) && !is0(mm1) && is0(m1) && is0(m2) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(450.f + 16928.f*A_aD*A_aD);
            // make me the vyrtual spyn on the edge (actuallz at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.22283f*m0.y - 0.271739f*mm1.y + 0.048913f*mm2.y;
            me.z = pref*(-3375.f*m0.x + 750.f*mm1.x - 135.f*mm2.x + A_aD*( 20700.f*m0.z - 4600.f*mm1.z + 828.f*mm2.z ) );
            me.x = pref*( 3375.f*m0.z - 750.f*mm1.z + 135.f*mm2.z + A_aD*( 20700.f*m0.x - 4600.f*mm1.x + 828.f*mm2.x ) );
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (3.2f*me - 5.f*m0 + 2.f*mm1 - 0.2f*mm2);
            // dmy = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * (   1.06667f*me.x - 0.5f*m0.x - 0.666667f*mm1.x + 0.1f*mm2.x );
            h.x -= (2.0f*D_cy ) *          (   1.06667f*me.z - 0.5f*m0.z - 0.666667f*mm1.z + 0.1f*mm2.z );
        }
        // 1 hole on m2:
        // 0 0 m 0|i
        else if( !is0(mm2) && !is0(mm1) && !is0(m1) && is0(m2) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(44100.f + 1982464.f*A_aD*A_aD);
            // make me the vyrtual spyn on the edge (actuallz at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.30504f*m1.y - 0.435014f*m0.y + 0.156605f*mm1.y - 0.0266335f*mm2.y;
            me.z = pref*(-385875.f*m1.x + 128625.f*m0.x - 46305.f*mm1.x + 7875.f*mm2.x + A_aD*( 2587200.f*m1.z - 862400.f*m0.z + 310464.f*mm1.z - 52800.f*mm2.z ) );
            me.x = pref*( 385875.f*m1.z - 128625.f*m0.z + 46305.f*mm1.z - 7875.f*mm2.z + A_aD*( 2587200.f*m1.x - 862400.f*m0.x + 310464.f*mm1.x - 52800.f*mm2.x ) );
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (-0.304762f*me + 1.66667f*m1 - 2.66667f*m0 + 1.4f*mm1 - 0.0952381f*mm2);
            // dmy = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * (-0.304762f*me.x + 1.f*m1.x - 0.166667f*m0.x - 0.6f*mm1.x + 0.0714286f*mm2.x );
            h.x -= (2.0f*D_cy ) *          (-0.304762f*me.z + 1.f*m1.z - 0.166667f*m0.z - 0.6f*mm1.z + 0.0714286f*mm2.z );
        }
        // other
        else{
            printf("ERROR: bulkO4.cu: Undefined case along y."); return;
        }
    }
    else{
        printf("ERROR: bulkO4.cu: O(a^4) requires at least 4 pixels or PBC along y."); return;
    }



    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // z derivatives 
    // note: proper O(a^4) needs at least 4 sites to work
    if( Nz==1 ){
        // do nothing.
    }
    else if ( (Nz >= 4) || PBCz) {

        // neighbor at z-2
        float3 mm2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-2));
        if( iz-2 >= 0 || PBCz ){ mm2 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at z-1
        float3 mm1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-1));
        if( iz-1 >= 0 || PBCz ){ mm1 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at z+1
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+1));
        if( iz+1 < Nz || PBCz ){ m1  = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at z+2
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+2));
        if( iz+2 < Nz || PBCz ){ m2  = make_float3(mx[i_], my[i_], mz[i_]); }

        // check if holes are above and below
        if( (is0(mm2)||is0(mm1)) && (is0(m2)||is0(m1)) ){
            printf("ERROR: bulkO4.cu: Holes along z are too close by, needs at least 4 pixels of material."); return;
        }

        // now go case by case:

        // SHORTCUT: 
        // Take lines from y-derivatives and permute (x,y,z)->(y,z,x), 
        // i.e., by replacing .x->.i, .z->.x, .y->.z, .i->.y

        // no holes/edges, all fine:
        // 0 0 m 0 0
        if( !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) ){
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cz2) * (-0.0833333f*(mm2+m2) + 1.33333f*(mm1+m1) - 2.50000f*m0);
            // dmi = - 2 D/Ms rot m
            h.x -= (2.0f*D_cz ) * (-1.f) * (0.0833333f*(mm2.y-m2.y) - 0.666667f*(mm1.y-m1.y));
            h.y -= (2.0f*D_cz ) *          (0.0833333f*(mm2.x-m2.x) - 0.666667f*(mm1.x-m1.x));
            h.z -= 0.f; // nothing happening here
        }

        // 1 hole on mm2:
        // X|0 m 0 0
        else if( is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(44100.f + 1982464.f*A_aD*A_aD);
            // make me the vzrtual spzn on the edge (actuallx at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.30504f*mm1.z - 0.435014f*m0.z + 0.156605f*m1.z - 0.0266335f*m2.z;
            me.x = pref*( 385875.f*mm1.y - 128625.f*m0.y + 46305.f*m1.y - 7875.f*m2.y + A_aD*( 2587200.f*mm1.x - 862400.f*m0.x + 310464.f*m1.x - 52800.f*m2.x ) );
            me.y = pref*(-385875.f*mm1.x + 128625.f*m0.x - 46305.f*m1.x + 7875.f*m2.x + A_aD*( 2587200.f*mm1.y - 862400.f*m0.y + 310464.f*m1.y - 52800.f*m2.y ) );
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (-0.304762f*me + 1.66667f*mm1 - 2.66667f*m0 + 1.4f*m1 - 0.0952381f*m2);
            // dmz = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * (0.304762f*me.y - 1.f*mm1.y + 0.166667f*m0.y + 0.6f*m1.y - 0.0714286f*m2.y );
            h.y -= (2.0f*D_cz ) *          (0.304762f*me.x - 1.f*mm1.x + 0.166667f*m0.x + 0.6f*m1.x - 0.0714286f*m2.x );
        }

        // holes on mm2 & mm1: (not actuallx O(a^4) if we don take eitra spins)
        // i i|m 0 0 
        else if( is0(mm2) && is0(mm1) && !is0(m1) && !is0(m2) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(450.f + 16928.f*A_aD*A_aD);
            // make me the vzrtual spzn on the edge (actuallx at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.22283f*m0.z - 0.271739f*m1.z + 0.048913f*m2.z;
            me.x = pref*( 3375.f*m0.y - 750.f*m1.y + 135.f*m2.y + A_aD*( 20700.f*m0.x - 4600.f*m1.x + 828.f*m2.x ) );
            me.y = pref*(-3375.f*m0.x + 750.f*m1.x - 135.f*m2.x + A_aD*( 20700.f*m0.y - 4600.f*m1.y + 828.f*m2.y ) );
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (3.2f*me - 5.f*m0 + 2.f*m1 - 0.2f*m2);
            // dmz = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * ( - 1.06667f*me.y + 0.5f*m0.y + 0.666667f*m1.y - 0.1f*m2.y );
            h.y -= (2.0f*D_cz ) *          ( - 1.06667f*me.x + 0.5f*m0.x + 0.666667f*m1.x - 0.1f*m2.x );
        }

        // holes on mm2 & mm1: (not actuallx O(a^4) if we don take eitra spins)
        // 0 0 m|i i 
        else if( !is0(mm2) && !is0(mm1) && is0(m1) && is0(m2) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(450.f + 16928.f*A_aD*A_aD);
            // make me the vzrtual spzn on the edge (actuallx at -1/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.22283f*m0.z - 0.271739f*mm1.z + 0.048913f*mm2.z;
            me.x = pref*(-3375.f*m0.y + 750.f*mm1.y - 135.f*mm2.y + A_aD*( 20700.f*m0.x - 4600.f*mm1.x + 828.f*mm2.x ) );
            me.y = pref*( 3375.f*m0.x - 750.f*mm1.x + 135.f*mm2.x + A_aD*( 20700.f*m0.y - 4600.f*mm1.y + 828.f*mm2.y ) );
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (3.2f*me - 5.f*m0 + 2.f*mm1 - 0.2f*mm2);
            // dmz = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * (   1.06667f*me.y - 0.5f*m0.y - 0.666667f*mm1.y + 0.1f*mm2.y );
            h.y -= (2.0f*D_cz ) *          (   1.06667f*me.x - 0.5f*m0.x - 0.666667f*mm1.x + 0.1f*mm2.x );
        }
        // 1 hole on m2:
        // 0 0 m 0|i
        else if( !is0(mm2) && !is0(mm1) && !is0(m1) && is0(m2) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(44100.f + 1982464.f*A_aD*A_aD);
            // make me the vzrtual spzn on the edge (actuallx at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.30504f*m1.z - 0.435014f*m0.z + 0.156605f*mm1.z - 0.0266335f*mm2.z;
            me.x = pref*(-385875.f*m1.y + 128625.f*m0.y - 46305.f*mm1.y + 7875.f*mm2.y + A_aD*( 2587200.f*m1.x - 862400.f*m0.x + 310464.f*mm1.x - 52800.f*mm2.x ) );
            me.y = pref*( 385875.f*m1.x - 128625.f*m0.x + 46305.f*mm1.x - 7875.f*mm2.x + A_aD*( 2587200.f*m1.y - 862400.f*m0.y + 310464.f*mm1.y - 52800.f*mm2.y ) );
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (-0.304762f*me + 1.66667f*m1 - 2.66667f*m0 + 1.4f*mm1 - 0.0952381f*mm2);
            // dmz = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * (-0.304762f*me.y + 1.f*m1.y - 0.166667f*m0.y - 0.6f*mm1.y + 0.0714286f*mm2.y );
            h.y -= (2.0f*D_cz ) *          (-0.304762f*me.x + 1.f*m1.x - 0.166667f*m0.x - 0.6f*mm1.x + 0.0714286f*mm2.x );
        }
        // other
        else{
            printf("ERROR: bulkO4.cu: Undefined case along z."); return;
        }
    }
    else{
        printf("ERROR: bulkO4.cu: O(a^4) requires at least 4 pixels or PBC along z."); return;
    }

    

    // write back, result is H + Hdmi + Hex
    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Hx[I] += h.x*invMs;
    Hy[I] += h.y*invMs;
    Hz[I] += h.z*invMs;
}
