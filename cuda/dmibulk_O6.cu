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
adddmibulko6(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
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
    // note: proper O(a^6) needs at least 6 sites to work
    if( Nx==1 ){
        // do nothing.
    }
    else if ( (Nx >= 6) || PBCx) {

        // neighbor at x-3
        float3 mm3 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(lclampx(ix-3), iy, iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( ix-3 >= 0 || PBCx ){ mm3 = make_float3(mx[i_], my[i_], mz[i_]); }
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
        // neighbor at x+3
        float3 m3 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(hclampx(ix+3), iy, iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( ix+3 < Nx || PBCx ){ m3  = make_float3(mx[i_], my[i_], mz[i_]); }

        // check if holes are above and below
        if( (is0(mm3)||is0(mm2)||is0(mm1)) && (is0(m3)||is0(m2)||is0(m1)) ){
            printf("ERROR: dmibulkO6.cu: Holes along x are too close by, needs at least 6 pixels of material."); return;
        }

        // now go case by case:

        // no holes/edges, all fine:
        // 0 0 0 m 0 0 0
        if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (0.0111111f*(mm3+m3) - 0.15f*(mm2+m2) + 1.5f*(mm1+m1) - 2.72222f*m0);
            // dmi = - 2 D/Ms rot m
            h.x -= 0.f; // nothing happening here
            h.y -= (2.0f*D_cx ) * (-1.f) * (-0.0166667f*(mm3.z-m3.z) + 0.15f*(mm2.z-m2.z) - 0.75f*(mm1.z-m1.z));
            h.z -= (2.0f*D_cx ) *          (-0.0166667f*(mm3.y-m3.y) + 0.15f*(mm2.y-m2.y) - 0.75f*(mm1.y-m1.y));
        }

        // hole on mm3:
        // X|0 0 m 0 0 0
        else if( is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(0.01771704983257333f + A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.44128f*mm2.x - 0.800712f*mm1.x + 0.576513f*m0.x - 0.294139f*m1.x + 0.088968f*m2.x - 0.0119114f*m3.x;
            me.y = pref*( 0.191842f*mm2.z - 0.106579f*mm1.z + 0.076737f*m0.z - 0.0391515f*m1.z + 0.0118421f*m2.z - 0.00158547f*m3.z + A_aD*( 1.44128f*mm2.y - 0.800712f*mm1.y + 0.576513f*m0.y - 0.294139f*m1.y + 0.088968f*m2.y - 0.0119114f*m3.y ));
            me.z = pref*(-0.191842f*mm2.y + 0.106579f*mm1.y - 0.076737f*m0.y + 0.0391515f*m1.y - 0.0118421f*m2.y + 0.00158547f*m3.y + A_aD*( 1.44128f*mm2.z - 0.800712f*mm1.z + 0.576513f*m0.z - 0.294139f*m1.z + 0.088968f*m2.z - 0.0119114f*m3.z ));
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (0.0492544f*me - 0.216667f*mm2 +  1.55556f*mm1- 2.76667f*m0 + 1.52381f*m1 - 0.157407f*m2 + 0.0121212f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * (-0.0738817f*me.z + 0.25f*mm2.z - 0.833333f*mm1.z + 0.0666667f*m0.z + 0.714286f*m1.z - 0.138889f*m2.z + 0.0151515f*m3.z );
            h.z -= (2.0f*D_cx ) *          (-0.0738817f*me.y + 0.25f*mm2.y - 0.833333f*mm1.y + 0.0666667f*m0.y + 0.714286f*m1.y - 0.138889f*m2.y + 0.0151515f*m3.y );
        }
        
        // hole on m3:
        // 0 0 0 m 0 0|X
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && is0(m3) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(0.01771704983257333f + A_aD*A_aD);
            // make me the virtual spin on the edge (actually at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.44128f*m2.x - 0.800712f*m1.x + 0.576513f*m0.x - 0.294139f*mm1.x + 0.088968f*mm2.x - 0.0119114f*mm3.x;
            me.y = pref*(-0.191842f*m2.z + 0.106579f*m1.z - 0.076737f*m0.z + 0.0391515f*mm1.z - 0.0118421f*mm2.z + 0.00158547f*mm3.z + A_aD*( 1.44128f*m2.y - 0.800712f*m1.y + 0.576513f*m0.y - 0.294139f*mm1.y + 0.088968f*mm2.y - 0.0119114f*mm3.y ));
            me.z = pref*( 0.191842f*m2.y - 0.106579f*m1.y + 0.076737f*m0.y - 0.0391515f*mm1.y + 0.0118421f*mm2.y - 0.00158547f*mm3.y + A_aD*( 1.44128f*m2.z - 0.800712f*m1.z + 0.576513f*m0.z - 0.294139f*mm1.z + 0.088968f*mm2.z - 0.0119114f*mm3.z ));
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (0.0492544f*me - 0.216667f*m2 +  1.55556f*m1- 2.76667f*m0 + 1.52381f*mm1 - 0.157407f*mm2 + 0.0121212f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * ( 0.0738817f*me.z - 0.25f*m2.z + 0.833333f*m1.z - 0.0666667f*m0.z - 0.714286f*mm1.z + 0.138889f*mm2.z - 0.0151515f*mm3.z );
            h.z -= (2.0f*D_cx ) *          ( 0.0738817f*me.y - 0.25f*m2.y + 0.833333f*m1.y - 0.0666667f*m0.y - 0.714286f*mm1.y + 0.138889f*mm2.y - 0.0151515f*mm3.y );
        }

        // hole on mm2:
        // ? X|0 m 0 0 0
        else if( is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(0.019565201959813105f + A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.3769f*mm1.x - 0.611956f*m0.x + 0.330456f*m1.x - 0.1124f*m2.x + 0.0169988f*m3.x;
            me.y = pref*( 0.192595f*mm1.z - 0.0855978f*m0.z + 0.0462228f*m1.z - 0.015722f*m2.z + 0.00237772f*m3.z + A_aD*( 1.3769f*mm1.y - 0.611956f*m0.y + 0.330456f*m1.y - 0.1124f*m2.y + 0.0169988f*m3.y ));
            me.z = pref*(-0.192595f*mm1.y + 0.0855978f*m0.y - 0.0462228f*m1.y + 0.015722f*m2.y - 0.00237772f*m3.y + A_aD*( 1.3769f*mm1.z - 0.611956f*m0.z + 0.330456f*m1.z - 0.1124f*m2.z + 0.0169988f*m3.z ));
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (-0.338624f*me + 1.75f*mm1 - 2.77778f*m0 + 1.5f*m1 - 0.142857f*m2 + 0.00925926f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * ( 0.203175f*me.z - 0.75f*mm1.z - 0.166667f*m0.z + 0.9f*m1.z - 0.214286f*m2.z + 0.0277778f*m3.z );
            h.z -= (2.0f*D_cx ) *          ( 0.203175f*me.y - 0.75f*mm1.y - 0.166667f*m0.y + 0.9f*m1.y - 0.214286f*m2.y + 0.0277778f*m3.y );
        }
        
        // hole on m2:
        // 0 0 0 m 0|X ?
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && is0(m2)  ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(0.019565201959813105f + A_aD*A_aD);
            // make me the virtual spin on the edge (actually at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = -1.3769f*m1.x + 0.611956f*m0.x - 0.330456f*mm1.x + 0.1124f*mm2.x - 0.0169988f*mm3.x;
            me.y = pref*(-0.192595f*m1.z + 0.0855978f*m0.z - 0.0462228f*mm1.z + 0.015722f*mm2.z - 0.00237772f*mm3.z + A_aD*( 1.3769f*m1.y - 0.611956f*m0.y + 0.330456f*mm1.y - 0.1124f*mm2.y + 0.0169988f*mm3.y ));
            me.z = pref*( 0.192595f*m1.y - 0.0855978f*m0.y + 0.0462228f*mm1.y - 0.015722f*mm2.y + 0.00237772f*mm3.y + A_aD*( 1.3769f*m1.z - 0.611956f*m0.z + 0.330456f*mm1.z - 0.1124f*mm2.z + 0.0169988f*mm3.z ));
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (-0.338624f*me + 1.75f*m1 - 2.77778f*m0 + 1.5f*mm1 - 0.142857f*mm2 + 0.00925926f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * (-0.203175f*me.z + 0.75f*m1.z + 0.166667f*m0.z - 0.9f*mm1.z + 0.214286f*mm2.z - 0.0277778f*mm3.z );
            h.z -= (2.0f*D_cx ) *          (-0.203175f*me.y + 0.75f*m1.y + 0.166667f*m0.y - 0.9f*mm1.y + 0.214286f*mm2.y - 0.0277778f*mm3.y );
        }

        // hole on mm1:
        // ? ? X|m 0 0 0
        else if( is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(0.022245044550619833f + A_aD*A_aD);
            // make me the virtual spin on the edge (actually at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = 1.30504f*m0.x - 0.435014f*m1.x + 0.156605f*m2.x - 0.0266335f*m3.x;
            me.y = pref*( 0.194644f*m0.z - 0.0648814f*m1.z + 0.0233573f*m2.z - 0.00397233f*m3.z + A_aD*( 1.30504f*m0.y - 0.435014f*m1.y + 0.156605f*m2.y - 0.0266335f*m3.y ));
            me.z = pref*(-0.194644f*m0.y + 0.0648814f*m1.y - 0.0233573f*m2.y + 0.00397233f*m3.y + A_aD*( 1.30504f*m0.z - 0.435014f*m1.z + 0.156605f*m2.z - 0.0266335f*m3.z ));           
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (3.35238f*me - 5.33333f*m0 + 2.33333f*m1 - 0.4f*m2 + 0.047619f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * (-0.914286f*me.z + 0.166667f*m0.z + 1.f*m1.z - 0.3f*m2.z + 0.047619f*m3.z );
            h.z -= (2.0f*D_cx ) *          (-0.914286f*me.y + 0.166667f*m0.y + 1.f*m1.y - 0.3f*m2.y + 0.047619f*m3.y );
        }
        
        // hole on m1:
        // 0 0 0 m|X ? ?
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && is0(m1) ){
            const float A_aD = A_cx2/D_cx;
            const float pref = A_aD/(0.022245044550619833f + A_aD*A_aD);
            // make me the virtual spin on the edge (actually at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.x = -1.30504f*m0.x + 0.435014f*mm1.x - 0.156605f*mm2.x + 0.0266335f*mm3.x;
            me.y = pref*(-0.194644f*m0.z + 0.0648814f*mm1.z - 0.0233573f*mm2.z + 0.00397233f*mm3.z + A_aD*( 1.30504f*m0.y - 0.435014f*mm1.y + 0.156605f*mm2.y - 0.0266335f*mm3.y ));
            me.z = pref*( 0.194644f*m0.y - 0.0648814f*mm1.y + 0.0233573f*mm2.y - 0.00397233f*mm3.y + A_aD*( 1.30504f*m0.z - 0.435014f*mm1.z + 0.156605f*mm2.z - 0.0266335f*mm3.z ));           
            // exchange = 2 Aex/Ms Laplace m
            h   += (2.0f*A_cx2) * (3.35238f*me - 5.33333f*m0 + 2.33333f*mm1 - 0.4f*mm2 + 0.047619f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmzdx, dmydx} 
            h.x -= 0.f;
            h.y -= (2.0f*D_cx ) * (-1.f) * ( 0.914286f*me.z - 0.166667f*m0.z - 1.f*mm1.z + 0.3f*mm2.z - 0.047619f*mm3.z );
            h.z -= (2.0f*D_cx ) *          ( 0.914286f*me.y - 0.166667f*m0.y - 1.f*mm1.y + 0.3f*mm2.y - 0.047619f*mm3.y );
        }

        // other
        else{
            printf("ERROR: dmibulkO6.cu: Undefined case along x."); return;
        }
    }
    else{
        printf("ERROR: dmibulkO6.cu: O(a^6) requires at least 6 pixels or PBC along x."); return;
    }



    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // y derivatives 
    // note: proper O(a^6) needs at least 6 sites to work
    if( Ny==1 ){
        // do nothing.
    }
    else if ( (Ny >= 6) || PBCy) {

        // neighbor at x-3
        float3 mm3 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-3), iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( iy-3 >= 0 || PBCy ){ mm3 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x-2
        float3 mm2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-2), iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( iy-2 >= 0 || PBCy ){ mm2 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x-1
        float3 mm1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( iy-1 >= 0 || PBCy ){ mm1 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+1
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( iy+1 < Ny || PBCy ){ m1  = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+2
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+2), iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( iy+2 < Ny || PBCy ){ m2  = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+3
        float3 m3 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+3), iz);            // load neighbor m if inside grid, keep 0 otherwise
        if( iy+3 < Ny || PBCy ){ m3  = make_float3(mx[i_], my[i_], mz[i_]); }

        // check if holes are above and below
        if( (is0(mm3)||is0(mm2)||is0(mm1)) && (is0(m3)||is0(m2)||is0(m1)) ){
            printf("ERROR: dmibulkO6.cu: Holes along y are too close by, needs at least 6 pixels of material."); return;
        }



        // SHORTCUT FOR THE CREATION OF THESE LINES USING THE X-CASE: 
        // Take lines from x-derivatives and permute (x,y,z)->(y,z,x), 
        // i.e., by replacing .x->.i, .z->.x, .y->.z, .i->.y



        // now go case bz case:

        // no holes/edges, all fine:
        // 0 0 0 m 0 0 0
        if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (0.0111111f*(mm3+m3) - 0.15f*(mm2+m2) + 1.5f*(mm1+m1) - 2.72222f*m0);
            // dmi = - 2 D/Ms rot m
            h.y -= 0.f; // nothing happening here
            h.z -= (2.0f*D_cy ) * (-1.f) * (-0.0166667f*(mm3.x-m3.x) + 0.15f*(mm2.x-m2.x) - 0.75f*(mm1.x-m1.x));
            h.x -= (2.0f*D_cy ) *          (-0.0166667f*(mm3.z-m3.z) + 0.15f*(mm2.z-m2.z) - 0.75f*(mm1.z-m1.z));
        }

        // hole on mm3:
        // y|0 0 m 0 0 0
        else if( is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(0.01771704983257333f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallz at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.44128f*mm2.y - 0.800712f*mm1.y + 0.576513f*m0.y - 0.294139f*m1.y + 0.088968f*m2.y - 0.0119114f*m3.y;
            me.z = pref*( 0.191842f*mm2.x - 0.106579f*mm1.x + 0.076737f*m0.x - 0.0391515f*m1.x + 0.0118421f*m2.x - 0.00158547f*m3.x + A_aD*( 1.44128f*mm2.z - 0.800712f*mm1.z + 0.576513f*m0.z - 0.294139f*m1.z + 0.088968f*m2.z - 0.0119114f*m3.z ));
            me.x = pref*(-0.191842f*mm2.z + 0.106579f*mm1.z - 0.076737f*m0.z + 0.0391515f*m1.z - 0.0118421f*m2.z + 0.00158547f*m3.z + A_aD*( 1.44128f*mm2.x - 0.800712f*mm1.x + 0.576513f*m0.x - 0.294139f*m1.x + 0.088968f*m2.x - 0.0119114f*m3.x ));
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (0.0492544f*me - 0.216667f*mm2 +  1.55556f*mm1- 2.76667f*m0 + 1.52381f*m1 - 0.157407f*m2 + 0.0121212f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * (-0.0738817f*me.x + 0.25f*mm2.x - 0.833333f*mm1.x + 0.0666667f*m0.x + 0.714286f*m1.x - 0.138889f*m2.x + 0.0151515f*m3.x );
            h.x -= (2.0f*D_cy ) *          (-0.0738817f*me.z + 0.25f*mm2.z - 0.833333f*mm1.z + 0.0666667f*m0.z + 0.714286f*m1.z - 0.138889f*m2.z + 0.0151515f*m3.z );
        }
        
        // hole on m3:
        // 0 0 0 m 0 0|y
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && is0(m3) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(0.01771704983257333f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallz at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.44128f*m2.y - 0.800712f*m1.y + 0.576513f*m0.y - 0.294139f*mm1.y + 0.088968f*mm2.y - 0.0119114f*mm3.y;
            me.z = pref*(-0.191842f*m2.x + 0.106579f*m1.x - 0.076737f*m0.x + 0.0391515f*mm1.x - 0.0118421f*mm2.x + 0.00158547f*mm3.x + A_aD*( 1.44128f*m2.z - 0.800712f*m1.z + 0.576513f*m0.z - 0.294139f*mm1.z + 0.088968f*mm2.z - 0.0119114f*mm3.z ));
            me.x = pref*( 0.191842f*m2.z - 0.106579f*m1.z + 0.076737f*m0.z - 0.0391515f*mm1.z + 0.0118421f*mm2.z - 0.00158547f*mm3.z + A_aD*( 1.44128f*m2.x - 0.800712f*m1.x + 0.576513f*m0.x - 0.294139f*mm1.x + 0.088968f*mm2.x - 0.0119114f*mm3.x ));
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (0.0492544f*me - 0.216667f*m2 +  1.55556f*m1- 2.76667f*m0 + 1.52381f*mm1 - 0.157407f*mm2 + 0.0121212f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * ( 0.0738817f*me.x - 0.25f*m2.x + 0.833333f*m1.x - 0.0666667f*m0.x - 0.714286f*mm1.x + 0.138889f*mm2.x - 0.0151515f*mm3.x );
            h.x -= (2.0f*D_cy ) *          ( 0.0738817f*me.z - 0.25f*m2.z + 0.833333f*m1.z - 0.0666667f*m0.z - 0.714286f*mm1.z + 0.138889f*mm2.z - 0.0151515f*mm3.z );
        }

        // hole on mm2:
        // ? y|0 m 0 0 0
        else if( is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(0.019565201959813105f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallz at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.3769f*mm1.y - 0.611956f*m0.y + 0.330456f*m1.y - 0.1124f*m2.y + 0.0169988f*m3.y;
            me.z = pref*( 0.192595f*mm1.x - 0.0855978f*m0.x + 0.0462228f*m1.x - 0.015722f*m2.x + 0.00237772f*m3.x + A_aD*( 1.3769f*mm1.z - 0.611956f*m0.z + 0.330456f*m1.z - 0.1124f*m2.z + 0.0169988f*m3.z ));
            me.x = pref*(-0.192595f*mm1.z + 0.0855978f*m0.z - 0.0462228f*m1.z + 0.015722f*m2.z - 0.00237772f*m3.z + A_aD*( 1.3769f*mm1.x - 0.611956f*m0.x + 0.330456f*m1.x - 0.1124f*m2.x + 0.0169988f*m3.x ));
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (-0.338624f*me + 1.75f*mm1 - 2.77778f*m0 + 1.5f*m1 - 0.142857f*m2 + 0.00925926f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * ( 0.203175f*me.x - 0.75f*mm1.x - 0.166667f*m0.x + 0.9f*m1.x - 0.214286f*m2.x + 0.0277778f*m3.x );
            h.x -= (2.0f*D_cy ) *          ( 0.203175f*me.z - 0.75f*mm1.z - 0.166667f*m0.z + 0.9f*m1.z - 0.214286f*m2.z + 0.0277778f*m3.z );
        }
        
        // hole on m2:
        // 0 0 0 m 0|y ?
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && is0(m2)  ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(0.019565201959813105f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallz at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = -1.3769f*m1.y + 0.611956f*m0.y - 0.330456f*mm1.y + 0.1124f*mm2.y - 0.0169988f*mm3.y;
            me.z = pref*(-0.192595f*m1.x + 0.0855978f*m0.x - 0.0462228f*mm1.x + 0.015722f*mm2.x - 0.00237772f*mm3.x + A_aD*( 1.3769f*m1.z - 0.611956f*m0.z + 0.330456f*mm1.z - 0.1124f*mm2.z + 0.0169988f*mm3.z ));
            me.x = pref*( 0.192595f*m1.z - 0.0855978f*m0.z + 0.0462228f*mm1.z - 0.015722f*mm2.z + 0.00237772f*mm3.z + A_aD*( 1.3769f*m1.x - 0.611956f*m0.x + 0.330456f*mm1.x - 0.1124f*mm2.x + 0.0169988f*mm3.x ));
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (-0.338624f*me + 1.75f*m1 - 2.77778f*m0 + 1.5f*mm1 - 0.142857f*mm2 + 0.00925926f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * (-0.203175f*me.x + 0.75f*m1.x + 0.166667f*m0.x - 0.9f*mm1.x + 0.214286f*mm2.x - 0.0277778f*mm3.x );
            h.x -= (2.0f*D_cy ) *          (-0.203175f*me.z + 0.75f*m1.z + 0.166667f*m0.z - 0.9f*mm1.z + 0.214286f*mm2.z - 0.0277778f*mm3.z );
        }

        // hole on mm1:
        // ? ? y|m 0 0 0
        else if( is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(0.022245044550619833f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallz at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = 1.30504f*m0.y - 0.435014f*m1.y + 0.156605f*m2.y - 0.0266335f*m3.y;
            me.z = pref*( 0.194644f*m0.x - 0.0648814f*m1.x + 0.0233573f*m2.x - 0.00397233f*m3.x + A_aD*( 1.30504f*m0.z - 0.435014f*m1.z + 0.156605f*m2.z - 0.0266335f*m3.z ));
            me.x = pref*(-0.194644f*m0.z + 0.0648814f*m1.z - 0.0233573f*m2.z + 0.00397233f*m3.z + A_aD*( 1.30504f*m0.x - 0.435014f*m1.x + 0.156605f*m2.x - 0.0266335f*m3.x ));           
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (3.35238f*me - 5.33333f*m0 + 2.33333f*m1 - 0.4f*m2 + 0.047619f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * (-0.914286f*me.x + 0.166667f*m0.x + 1.f*m1.x - 0.3f*m2.x + 0.047619f*m3.x );
            h.x -= (2.0f*D_cy ) *          (-0.914286f*me.z + 0.166667f*m0.z + 1.f*m1.z - 0.3f*m2.z + 0.047619f*m3.z );
        }
        
        // hole on m1:
        // 0 0 0 m|y ? ?
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && is0(m1) ){
            const float A_aD = A_cy2/D_cy;
            const float pref = A_aD/(0.022245044550619833f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallz at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.y = -1.30504f*m0.y + 0.435014f*mm1.y - 0.156605f*mm2.y + 0.0266335f*mm3.y;
            me.z = pref*(-0.194644f*m0.x + 0.0648814f*mm1.x - 0.0233573f*mm2.x + 0.00397233f*mm3.x + A_aD*( 1.30504f*m0.z - 0.435014f*mm1.z + 0.156605f*mm2.z - 0.0266335f*mm3.z ));
            me.x = pref*( 0.194644f*m0.z - 0.0648814f*mm1.z + 0.0233573f*mm2.z - 0.00397233f*mm3.z + A_aD*( 1.30504f*m0.x - 0.435014f*mm1.x + 0.156605f*mm2.x - 0.0266335f*mm3.x ));           
            // eychange = 2 Aey/Ms Laplace m
            h   += (2.0f*A_cy2) * (3.35238f*me - 5.33333f*m0 + 2.33333f*mm1 - 0.4f*mm2 + 0.047619f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmxdy, dmzdy} 
            h.y -= 0.f;
            h.z -= (2.0f*D_cy ) * (-1.f) * ( 0.914286f*me.x - 0.166667f*m0.x - 1.f*mm1.x + 0.3f*mm2.x - 0.047619f*mm3.x );
            h.x -= (2.0f*D_cy ) *          ( 0.914286f*me.z - 0.166667f*m0.z - 1.f*mm1.z + 0.3f*mm2.z - 0.047619f*mm3.z );
        }

        // other
        else{
            printf("ERROR: dmibulkO6.cu: Undefined case along y."); return;
        }
    }
    else{
        printf("ERROR: dmibulkO6.cu: O(a^6) requires at least 6 piyels or PBC along y."); return;
    }



    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // z derivatives 
    // note: proper O(a^6) needs at least 6 sites to work
    if( Nz==1 ){
        // do nothing.
    }
    else if ( (Nz >= 6) || PBCz) {

        // neighbor at x-3
        float3 mm3 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-3));            // load neighbor m if inside grid, keep 0 otherwise
        if( iz-3 >= 0 || PBCz ){ mm3 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x-2
        float3 mm2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-2));            // load neighbor m if inside grid, keep 0 otherwise
        if( iz-2 >= 0 || PBCz ){ mm2 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x-1
        float3 mm1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-1));            // load neighbor m if inside grid, keep 0 otherwise
        if( iz-1 >= 0 || PBCz ){ mm1 = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+1
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+1));            // load neighbor m if inside grid, keep 0 otherwise
        if( iz+1 < Nz || PBCz ){ m1  = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+2
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+2));            // load neighbor m if inside grid, keep 0 otherwise
        if( iz+2 < Nz || PBCz ){ m2  = make_float3(mx[i_], my[i_], mz[i_]); }
        // neighbor at x+3
        float3 m3 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+3));            // load neighbor m if inside grid, keep 0 otherwise
        if( iz+3 < Nz || PBCz ){ m3  = make_float3(mx[i_], my[i_], mz[i_]); }

        // check if holes are above and below
        if( (is0(mm3)||is0(mm2)||is0(mm1)) && (is0(m3)||is0(m2)||is0(m1)) ){
            printf("ERROR: dmibulkO6.cu: Holes along z are too close by, needs at least 6 pixels of material."); return;
        }



        // SHORTCUT FOR THE CREATION OF THESE LINES USING THE Y-CASE: 
        // Take lines from y-derivatives and permute (x,y,z)->(y,z,x), 
        // i.e., by replacing .x->.i, .z->.x, .y->.z, .i->.y



        // now go case bx case:

        // no holes/edges, all fine:
        // 0 0 0 m 0 0 0
        if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (0.0111111f*(mm3+m3) - 0.15f*(mm2+m2) + 1.5f*(mm1+m1) - 2.72222f*m0);
            // dmi = - 2 D/Ms rot m
            h.z -= 0.f; // nothing happening here
            h.x -= (2.0f*D_cz ) * (-1.f) * (-0.0166667f*(mm3.y-m3.y) + 0.15f*(mm2.y-m2.y) - 0.75f*(mm1.y-m1.y));
            h.y -= (2.0f*D_cz ) *          (-0.0166667f*(mm3.x-m3.x) + 0.15f*(mm2.x-m2.x) - 0.75f*(mm1.x-m1.x));
        }

        // hole on mm3:
        // z|0 0 m 0 0 0
        else if( is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(0.01771704983257333f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallx at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.44128f*mm2.z - 0.800712f*mm1.z + 0.576513f*m0.z - 0.294139f*m1.z + 0.088968f*m2.z - 0.0119114f*m3.z;
            me.x = pref*( 0.191842f*mm2.y - 0.106579f*mm1.y + 0.076737f*m0.y - 0.0391515f*m1.y + 0.0118421f*m2.y - 0.00158547f*m3.y + A_aD*( 1.44128f*mm2.x - 0.800712f*mm1.x + 0.576513f*m0.x - 0.294139f*m1.x + 0.088968f*m2.x - 0.0119114f*m3.x ));
            me.y = pref*(-0.191842f*mm2.x + 0.106579f*mm1.x - 0.076737f*m0.x + 0.0391515f*m1.x - 0.0118421f*m2.x + 0.00158547f*m3.x + A_aD*( 1.44128f*mm2.y - 0.800712f*mm1.y + 0.576513f*m0.y - 0.294139f*m1.y + 0.088968f*m2.y - 0.0119114f*m3.y ));
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (0.0492544f*me - 0.216667f*mm2 +  1.55556f*mm1- 2.76667f*m0 + 1.52381f*m1 - 0.157407f*m2 + 0.0121212f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * (-0.0738817f*me.y + 0.25f*mm2.y - 0.833333f*mm1.y + 0.0666667f*m0.y + 0.714286f*m1.y - 0.138889f*m2.y + 0.0151515f*m3.y );
            h.y -= (2.0f*D_cz ) *          (-0.0738817f*me.x + 0.25f*mm2.x - 0.833333f*mm1.x + 0.0666667f*m0.x + 0.714286f*m1.x - 0.138889f*m2.x + 0.0151515f*m3.x );
        }
        
        // hole on m3:
        // 0 0 0 m 0 0|z
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && is0(m3) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(0.01771704983257333f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallx at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.44128f*m2.z - 0.800712f*m1.z + 0.576513f*m0.z - 0.294139f*mm1.z + 0.088968f*mm2.z - 0.0119114f*mm3.z;
            me.x = pref*(-0.191842f*m2.y + 0.106579f*m1.y - 0.076737f*m0.y + 0.0391515f*mm1.y - 0.0118421f*mm2.y + 0.00158547f*mm3.y + A_aD*( 1.44128f*m2.x - 0.800712f*m1.x + 0.576513f*m0.x - 0.294139f*mm1.x + 0.088968f*mm2.x - 0.0119114f*mm3.x ));
            me.y = pref*( 0.191842f*m2.x - 0.106579f*m1.x + 0.076737f*m0.x - 0.0391515f*mm1.x + 0.0118421f*mm2.x - 0.00158547f*mm3.x + A_aD*( 1.44128f*m2.y - 0.800712f*m1.y + 0.576513f*m0.y - 0.294139f*mm1.y + 0.088968f*mm2.y - 0.0119114f*mm3.y ));
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (0.0492544f*me - 0.216667f*m2 +  1.55556f*m1- 2.76667f*m0 + 1.52381f*mm1 - 0.157407f*mm2 + 0.0121212f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * ( 0.0738817f*me.y - 0.25f*m2.y + 0.833333f*m1.y - 0.0666667f*m0.y - 0.714286f*mm1.y + 0.138889f*mm2.y - 0.0151515f*mm3.y );
            h.y -= (2.0f*D_cz ) *          ( 0.0738817f*me.x - 0.25f*m2.x + 0.833333f*m1.x - 0.0666667f*m0.x - 0.714286f*mm1.x + 0.138889f*mm2.x - 0.0151515f*mm3.x );
        }

        // hole on mm2:
        // ? z|0 m 0 0 0
        else if( is0(mm2) && !is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(0.019565201959813105f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallx at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.3769f*mm1.z - 0.611956f*m0.z + 0.330456f*m1.z - 0.1124f*m2.z + 0.0169988f*m3.z;
            me.x = pref*( 0.192595f*mm1.y - 0.0855978f*m0.y + 0.0462228f*m1.y - 0.015722f*m2.y + 0.00237772f*m3.y + A_aD*( 1.3769f*mm1.x - 0.611956f*m0.x + 0.330456f*m1.x - 0.1124f*m2.x + 0.0169988f*m3.x ));
            me.y = pref*(-0.192595f*mm1.x + 0.0855978f*m0.x - 0.0462228f*m1.x + 0.015722f*m2.x - 0.00237772f*m3.x + A_aD*( 1.3769f*mm1.y - 0.611956f*m0.y + 0.330456f*m1.y - 0.1124f*m2.y + 0.0169988f*m3.y ));
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (-0.338624f*me + 1.75f*mm1 - 2.77778f*m0 + 1.5f*m1 - 0.142857f*m2 + 0.00925926f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * ( 0.203175f*me.y - 0.75f*mm1.y - 0.166667f*m0.y + 0.9f*m1.y - 0.214286f*m2.y + 0.0277778f*m3.y );
            h.y -= (2.0f*D_cz ) *          ( 0.203175f*me.x - 0.75f*mm1.x - 0.166667f*m0.x + 0.9f*m1.x - 0.214286f*m2.x + 0.0277778f*m3.x );
        }
        
        // hole on m2:
        // 0 0 0 m 0|z ?
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && !is0(m1) && is0(m2)  ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(0.019565201959813105f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallx at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = -1.3769f*m1.z + 0.611956f*m0.z - 0.330456f*mm1.z + 0.1124f*mm2.z - 0.0169988f*mm3.z;
            me.x = pref*(-0.192595f*m1.y + 0.0855978f*m0.y - 0.0462228f*mm1.y + 0.015722f*mm2.y - 0.00237772f*mm3.y + A_aD*( 1.3769f*m1.x - 0.611956f*m0.x + 0.330456f*mm1.x - 0.1124f*mm2.x + 0.0169988f*mm3.x ));
            me.y = pref*( 0.192595f*m1.x - 0.0855978f*m0.x + 0.0462228f*mm1.x - 0.015722f*mm2.x + 0.00237772f*mm3.x + A_aD*( 1.3769f*m1.y - 0.611956f*m0.y + 0.330456f*mm1.y - 0.1124f*mm2.y + 0.0169988f*mm3.y ));
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (-0.338624f*me + 1.75f*m1 - 2.77778f*m0 + 1.5f*mm1 - 0.142857f*mm2 + 0.00925926f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * (-0.203175f*me.y + 0.75f*m1.y + 0.166667f*m0.y - 0.9f*mm1.y + 0.214286f*mm2.y - 0.0277778f*mm3.y );
            h.y -= (2.0f*D_cz ) *          (-0.203175f*me.x + 0.75f*m1.x + 0.166667f*m0.x - 0.9f*mm1.x + 0.214286f*mm2.x - 0.0277778f*mm3.x );
        }

        // hole on mm1:
        // ? ? z|m 0 0 0
        else if( is0(mm1) && !is0(m1) && !is0(m2) && !is0(m3) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(0.022245044550619833f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallx at -3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = 1.30504f*m0.z - 0.435014f*m1.z + 0.156605f*m2.z - 0.0266335f*m3.z;
            me.x = pref*( 0.194644f*m0.y - 0.0648814f*m1.y + 0.0233573f*m2.y - 0.00397233f*m3.y + A_aD*( 1.30504f*m0.x - 0.435014f*m1.x + 0.156605f*m2.x - 0.0266335f*m3.x ));
            me.y = pref*(-0.194644f*m0.x + 0.0648814f*m1.x - 0.0233573f*m2.x + 0.00397233f*m3.x + A_aD*( 1.30504f*m0.y - 0.435014f*m1.y + 0.156605f*m2.y - 0.0266335f*m3.y ));           
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (3.35238f*me - 5.33333f*m0 + 2.33333f*m1 - 0.4f*m2 + 0.047619f*m3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * (-0.914286f*me.y + 0.166667f*m0.y + 1.f*m1.y - 0.3f*m2.y + 0.047619f*m3.y );
            h.y -= (2.0f*D_cz ) *          (-0.914286f*me.x + 0.166667f*m0.x + 1.f*m1.x - 0.3f*m2.x + 0.047619f*m3.x );
        }
        
        // hole on m1:
        // 0 0 0 m|z ? ?
        else if( !is0(mm3) && !is0(mm2) && !is0(mm1) && is0(m1) ){
            const float A_aD = A_cz2/D_cz;
            const float pref = A_aD/(0.022245044550619833f + A_aD*A_aD);
            // make me the virtual spin on the edge (actuallx at +3/2)
            float3 me = make_float3(0.0f, 0.0f, 0.0f);
            me.z = -1.30504f*m0.z + 0.435014f*mm1.z - 0.156605f*mm2.z + 0.0266335f*mm3.z;
            me.x = pref*(-0.194644f*m0.y + 0.0648814f*mm1.y - 0.0233573f*mm2.y + 0.00397233f*mm3.y + A_aD*( 1.30504f*m0.x - 0.435014f*mm1.x + 0.156605f*mm2.x - 0.0266335f*mm3.x ));
            me.y = pref*( 0.194644f*m0.x - 0.0648814f*mm1.x + 0.0233573f*mm2.x - 0.00397233f*mm3.x + A_aD*( 1.30504f*m0.y - 0.435014f*mm1.y + 0.156605f*mm2.y - 0.0266335f*mm3.y ));           
            // ezchange = 2 Aez/Ms Laplace m
            h   += (2.0f*A_cz2) * (3.35238f*me - 5.33333f*m0 + 2.33333f*mm1 - 0.4f*mm2 + 0.047619f*mm3 );
            // dmi = - 2 D/Ms rot m = - 2 D/Ms {0, -dmydz, dmxdz} 
            h.z -= 0.f;
            h.x -= (2.0f*D_cz ) * (-1.f) * ( 0.914286f*me.y - 0.166667f*m0.y - 1.f*mm1.y + 0.3f*mm2.y - 0.047619f*mm3.y );
            h.y -= (2.0f*D_cz ) *          ( 0.914286f*me.x - 0.166667f*m0.x - 1.f*mm1.x + 0.3f*mm2.x - 0.047619f*mm3.x );
        }

        // other
        else{
            printf("ERROR: dmibulkO6.cu: Undefined case along z."); return;
        }
    }
    else{
        printf("ERROR: dmibulkO6.cu: O(a^6) requires at least 6 pizels or PBC along z."); return;
    }

    

    // write back, result is H + Hdmi + Hex
    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Hx[I] += h.x*invMs;
    Hy[I] += h.y*invMs;
    Hz[I] += h.z*invMs;
}
