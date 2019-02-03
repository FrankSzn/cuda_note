#include<algorithm>

#define MAX_FRONTIER_SIZE 5


void insertInfoFrontier(int vertex, int* frontier, int* frontierSize){
	frontier[*frontierSize] = vertex;
	++(*frontierSize);
}

void BFS_sequential(int source, const int* rowPointers, const int* destination,
			int* distance)
{
	int frontier[2][MAX_FRONTIER_SIZE];
	int* currentFrontier = &frontier[0];
	int currentFrontierSize = 0;
	int* previousFrontier = &frontier[1];
	int preiousFrontierSize = 0;
	
	insertIntoFrontier(source, previousFrontier, &previousFrontierSize);
	distances[source] = 0;

	while(previousFrontierSize > 0){
		// visit all vertices on the previous frontier
		for(int f = 0; f < previousFrontierSize; f++){
			const int currentVertex = previousFrontier[f];
			// check all outgoing edges
			for(int i = rowPointers[currentVertex]; i < rowPointers[currentVertex+1]; ++i)
			{
				if(distances[destinations[i]] == -1)	{
					// this vertex has not been visited yet
					insertIntoFrontier(destinations[i], currentFrontier, &currentFrontierSize);
					distances[destinations[i]] = distances[currentVertex] + 1;
			}
		}
	}
	std::swap(currentFrontier, previousFrontier);
	previousFrontierSize = currentFrontierSize;
	currentFrontierSize = 0;
	
}
