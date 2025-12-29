#include <arm_neon.h>
#include <iostream>

using namespace std;

int main() {
  cout << "NeonFlux: Checking ARM64 NEON support..." << endl;
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float32x4_t vec = vld1q_f32(data);
  float32x4_t result = vaddq_f32(vec, vec);
  float out[4];
  vst1q_f32(out, result);

  cout << "Input:  {1.0, 2.0, 3.0, 4.0}" << endl;
  cout << "Result: {" << out[0] << ", " << out[1] << ", " << out[2] << ", "
       << out[3] << "}" << endl;

  if (out[0] == 2.0f && out[3] == 8.0f) {
    cout << "SUCCESS: NEON intrinsics are working!" << endl;
    return 0;
  } else {
    cout << "FAILURE: Arithmetic check failed." << endl;
    return 1;
  }
}
