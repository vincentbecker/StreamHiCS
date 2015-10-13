package centroids;

import java.util.Random;

import org.apache.commons.math3.util.MathArrays;

import misc.Callback;
import statisticalTests.StatisticsBundle;
import weka.core.Instance;

public class FixedCentroids extends CentroidsContainer {

	private Centroid[] centroids;
	private int numberOfCentroids;
	private int numberOfDimensions;
	private int centroidsPerDimension;
	private double[] stepWidth;
	/**
	 * A generator for random numbers.
	 */
	private Random generator;
	private int k;
	private double[] kNNRank;
	private double allowedChange;
	private double checkCount;
	Callback callback;

	public FixedCentroids(Callback callback, int centroidsPerDimension, double[] lowerBounds, double[] upperBounds,
			int checkCount, int k, double allowedChange) {
		// TODO: Argument checking
		this.centroidsPerDimension = centroidsPerDimension;
		this.numberOfDimensions = lowerBounds.length;
		this.numberOfCentroids = centroidsPerDimension * lowerBounds.length;
		this.centroids = new Centroid[numberOfCentroids];
		this.stepWidth = new double[numberOfDimensions];
		// Calculating the stepWidth for each dimension
		for (int i = 0; i < numberOfDimensions; i++) {
			stepWidth[i] = (upperBounds[i] - lowerBounds[i]) / centroidsPerDimension;
		}
		generator = new Random();
		this.checkCount = checkCount;
		this.k = k;
		this.allowedChange = allowedChange;
		this.callback = callback;
	}

	@Override
	public void add(Instance instance) {
		// Create vector
		double[] vector = new double[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			vector[i] = instance.value(i);
		}

		// Find nearest centroid.
		int index = mapping(vector);
		Centroid c = centroids[index];
		if (c == null) {
			c = new Centroid(createVectorFromIndex(index));
		}
		c.increment();
	}

	// Simple, because centroids are static.
	private int mapping(double[] vector) {
		int[] indexes = mappingToIndexes(vector);
		// Create internal index
		return internalIndex(indexes);
	}

	private int[] mappingToIndexes(double[] vector) {
		int step = 0;
		int[] indexes = new int[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			step = (int) (vector[0] / stepWidth[0]);
			if (step < 0) {
				step = 0;
			} else if (step >= centroidsPerDimension) {
				step = centroidsPerDimension - 1;
			}
			indexes[i] = step;
		}
		return indexes;
	}

	public Centroid getCentroid(int[] indexes) {
		int internalIndex = internalIndex(indexes);
		return centroids[internalIndex];
	}

	private int internalIndex(int[] indexes) {
		int internalIndex = 0;
		for (int i = 0; i < numberOfDimensions; i++) {
			internalIndex += indexes[i] * Math.pow(centroidsPerDimension, numberOfDimensions - i - 1);
		}
		return internalIndex;
	}

	private int[] indexesRepresentation(int internalIndex) {
		int[] indexes = new int[numberOfDimensions];
		int a;
		// Calculating the cell for each dimension
		for (int i = 0; i < numberOfDimensions; i++) {
			a = (int) Math.pow(centroidsPerDimension, numberOfDimensions - i - 1);
			indexes[i] = internalIndex / a;
			internalIndex = internalIndex % a;
		}

		return indexes;

	}

	private double[] createVectorFromIndex(int index) {
		double[] vector = new double[numberOfDimensions];
		int a;
		int step;
		// Calculating the cell for each dimension
		for (int i = 0; i < numberOfDimensions; i++) {
			a = (int) Math.pow(centroidsPerDimension, numberOfDimensions - i - 1);
			step = index / a;
			// Centroid is in the middle of the cell
			vector[i] = (step + 0.5) * stepWidth[i];
			index = index % a;
		}
		return vector;
	}

	@Override
	public void clear() {
		for (Centroid centroid : centroids) {
			centroid.setCount(0);
		}
	}

	@Override
	public int getNumberOfInstances() {
		return numberOfCentroids;
	}

	@Override
	public StatisticsBundle getProjectedDataStatistics(int referenceDimension) {
		return calculateStatistics(centroids, referenceDimension);
	}

	// Input is the array of shuffled dimensions
	@Override
	public StatisticsBundle getSlicedDataStatistics(int[] shuffledDimensions, double selectionAlpha) {
		// An array showing the bounds on centroids selected (lower bound on
		// centroid number and upperBound (both inclusive)).
		int[][] selection = new int[2][numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			selection[0][i] = 0;
			selection[1][i] = centroidsPerDimension - 1;
		}

		int r = 0;
		int actualDim = 0;
		int currentCount = 0;
		int selectionAmount = 0;
		for (int i = 0; i < shuffledDimensions.length - 1; i++) {
			actualDim = shuffledDimensions[actualDim];
			currentCount = countSelected(selection);
			selectionAmount = (int) selectionAlpha * currentCount;
			// Getting random starting point
			r = generator.nextInt(centroidsPerDimension);
			selection[0][actualDim] = r;
			selection[1][actualDim] = r;
			currentCount = countSelected(selection);
			// Growing block until selection criterion fulfilled
			while (currentCount < selectionAmount && selection[0][actualDim] >= 0
					&& selection[1][actualDim] < centroidsPerDimension) {
				r = generator.nextInt(2);
				if (r == 0) {
					if (selection[0][actualDim] > 0) {
						selection[0][actualDim]--;
					} else if (selection[1][actualDim] < centroidsPerDimension - 1) {
						selection[1][actualDim]++;
					}
				} else {
					if (selection[1][actualDim] < centroidsPerDimension - 1) {
						selection[1][actualDim]++;
					} else if (selection[0][actualDim] > 0) {
						selection[0][actualDim]--;
					}
				}
				currentCount = countSelected(selection);
			}
		}

		// The selection array defines the slice
		int referenceDimension = shuffledDimensions[shuffledDimensions.length];
		Centroid[] slice = new Centroid[currentCount];
		int index = 0;
		// Check every centroid if it is in the final selection, i.e. bounds
		int[] lowerBounds = selection[0];
		int[] upperBounds = selection[1];
		Centroid c;
		for (int i = 0; i < centroids.length; i++) {
			c = centroids[i];
			if (c != null && checkBounds(lowerBounds, upperBounds, i)) {
				slice[index] = c;
			}
		}

		return calculateStatistics(slice, referenceDimension);
	}

	private int countSelected(int[][] selection) {
		int totalCount = 0;
		int[] lowerBounds = selection[0];
		int[] upperBounds = selection[1];
		Centroid c;
		for (int i = 0; i < centroids.length; i++) {
			c = centroids[i];
			if (c != null && checkBounds(lowerBounds, upperBounds, i)) {
				totalCount += c.getCount();
			}
		}

		return totalCount;
	}

	private boolean checkBounds(int[] lowerBounds, int[] upperBounds, int internalIndex) {
		int a;
		int step;

		for (int i = 0; i < numberOfDimensions; i++) {
			a = (int) Math.pow(centroidsPerDimension, numberOfDimensions - i - 1);
			step = internalIndex / a;
			if (step < lowerBounds[i] || step > upperBounds[i]) {
				// Outside of bounds
				return false;
			}
		}
		return true;
	}

	public void densityCheck() {
		// Calculate kNN-distances and sort them.
		double[] kNNDistances = new double[numberOfCentroids];
		double[] indexes = new double[numberOfCentroids];

		for (int i = 0; i < numberOfCentroids; i++) {
			kNNDistances[i] = calculateKNNDistance(i);
			indexes[i] = i;
		}

		MathArrays.sortInPlace(kNNDistances, indexes);

		// Calculate the change in the kNN rank
		int change = 0;
		if (kNNRank != null) {
			for (int i = 0; i < numberOfCentroids; i++) {
				for (int j = 0; j < numberOfCentroids; j++) {
					if (kNNRank[i] == indexes[j]) {
						change += Math.abs(i - j);
					}
				}
			}
		}

		kNNRank = indexes;

		if (change > allowedChange * numberOfCentroids) {
			// Notify callback to build check contrast
			callback.onAlarm();
		}

	}

	private double calculateKNNDistance(int index) {

		Centroid c = centroids[index];
		/*
		 * //Get the indexes representation int[] indexes =
		 * indexesRepresentation(index); int[] distFactor = new
		 * int[numberOfDimensions];
		 * 
		 * int total = 0; double maxDistance = 0; double min = Double.MAX_VALUE;
		 * int minIndex = 0; Centroid c2; double d = 0;
		 * 
		 * while (total < k) { // Search for the minimum distance to a neighbour
		 * min = Double.MAX_VALUE; for (int i = 0; i < numberOfDimensions; i++)
		 * { d = distFactor[i]*stepWidth[i]; if (d < min) { min = d; minIndex =
		 * i; } }
		 * 
		 * //Get the upper and lower neighbour and add their counts
		 * distFactor[minIndex]++; indexes[minIndex] += distFactor[minIndex]; c2
		 * = centroids[internalIndex(indexes)]; total += c2.getCount();
		 * 
		 * vector[minIndex] -= 2*min;
		 * 
		 * maxDistance = min; dist[minIndex] += stepWidth[minIndex]; }
		 * 
		 * return maxDistance;
		 * 
		 */

		double[] kDistances = new double[k];
		for (int i = 0; i < k; i++) {
			kDistances[i] = Double.MAX_VALUE;
		}
		double max = -1;
		int maxIndex = 0;

		for (int i = 0; i < centroids.length; i++) {
			if (i != index) {
				max = -1;
				// Search for the maximum distance
				for (int j = 0; j < k; j++) {
					if (kDistances[j] > max) {
						max = kDistances[j];
						maxIndex = j;
					}
				}
				double currentDistance = euclideanCentroidDistance(c, centroids[i]);
				// Replace the maximum distance if applicable
				if (currentDistance < max) {
					kDistances[maxIndex] = currentDistance;
				}
			}
		}

		max = -1;
		// Search for the maximum distance
		for (int j = 0; j < k; j++) {
			if (kDistances[j] > max) {
				max = kDistances[j];
			}
		}
		return max;
	}

}
