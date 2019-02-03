
__global__ void convolution_2D_basic_kernel(unsigned char* in, unsigned char* mask, unsigned char* out, int maskwidth, int w, int h)
{
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	if(Col < w && Row < h){
		int pixVal = 0;

		N_start_col = Col - (maskwidth/2);
		N_start_row = Row - (maskwidth/2);

		// Get the of the surrounding box
		for(int j = 0; j < maskwidth; ++j){
			for(int k = 0; k < maskwidth; ++k){
				int curRow = N_Start_row + j;
				int curCol = N_start_col + k;
				// Verify we have a valid image pixel
				if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
					pixVal += in[curRow * w + curCol] * mask[j * maskwidth + k];
				}
			}
		}
		// Write out new pixel value out
		out[Row * w + Col] = (unsigned char)(pixVal);
	}
}
