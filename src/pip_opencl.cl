
__kernel void pip_naive(
   const uint n_points,
   const uint n_polygons,
   __global float2 *points,
   __global float4 *boxes,
   __global uint2 *polys,
   __global float4 *edges,
   __global int *result) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  if ((i < n_points) && (j < n_polygons)) {
    int cn = 0;
    float2 p = points[i];
    float2 mn = boxes[j].xy;
    float2 mx = boxes[j].zw;
    if (p.x > mn.x && p.x < mx.x && p.y > mn.y && p.y < mx.y) {
      for (int k = 0; k < polys[j].y; k++) {
        float2 e1 = edges[polys[j].x + k].xy;
        float2 e2 = edges[polys[j].x + k].zw;
        int c1 =
            ((e1.y <= p.y) && (e2.y > p.y)) || ((e1.y > p.y) && (e2.y <= p.y));
        float vt = (p.y - e1.y) / (e2.y - e1.y);
        int c2 = p.x < (e1.x + vt * (e2.x - e1.x));
        cn += c1 && c2;
      }
      if (cn & 1)
         result[i] = result[i] != -1 ? result[i] : ((int)(cn & 1)) * j;
    }
  }
}
