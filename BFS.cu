


#define MAX_FRONTIER_SIZE 5

void BFS_host(int source, const int* rowPointers, const int* destinations, int* distances)
{
	int dFrontier[2][MAX_FRONTIER_SIZE];
	int* dCurrentFrontierSize;
	int* dPreviousFrontierSize;
	int* hPreviousFrontierSize;
	int* dVisited;

	int* dCurrentFrontier = &frontier[0];
	int* dPreviousFrontier = &frontier[1];

	// allocate device memory, copy memory from device to host, initialize
	// frontier sizes, etc.
	...
	
	hPreviousFrontierSize = 1;
	while(hPrevious)
}
