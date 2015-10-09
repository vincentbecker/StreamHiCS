package centroids;

import java.util.Random;

import org.apache.commons.math3.util.MathArrays;

import statisticalTests.StatisticsBundle;
import weka.core.Instance;

public class FixedCentroids extends CentroidsContainer {

	private Centroid[] centroids;
	private int numberOfCentroids;
	private int numberOfDimensions;
	private int centroidsPerDimension;
	private double[] stepWidth;
	private double selectionAlpha;
	/**
	 * A generator for random numbers.
	 */
	private Random generator;
	private int k;
	private double[] kNNRank;
	private double allowedChange;
	private double checkCount;

	public FixedCentroids(int centroidsPerDimension, double[] lowerBounds, double[] upperBounds, double selectionAlpha,
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
		this.selectionAlpha = selectionAlpha;
		generator = new Random();
		this.checkCount = checkCount;
		this.k = k;
		this.allowedChange = allowedChange;
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
		// Create internal index
		return internalIndex(indexes);
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
	public StatisticsBundle getProjectedDataStaistics(int referenceDimension) {
		return calculateStatistics(centroids, referenceDimension);
	}

	// Input is the array of shuffled dimensions
	@Override
	public StatisticsBundle getSlicedDataStaistics(int[] shuffledDimensions) {
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

	private StatisticsBundle calculateStatistics(Centroid[] centroidSelection, int referenceDimension) {
		int totalCount = 0;
		double totalSum = 0;
		int count = 0;
		double mean = 0;
		double variance = 0;

		// Calculation of mean
		for (Centroid centroid : centroidSelection) {
			if (centroid != null) {
				count = centroid.getCount();
				totalSum += centroid.getVector()[referenceDimension] * count;
				totalCount += count;
			}
		}
		mean = totalSum / totalCount;

		// Calculation of variance
		totalSum = 0;
		for (Centroid centroid : centroidSelection) {
			if (centroid != null) {
				totalSum += centroid.getCount() * Math.pow(centroid.getVector()[referenceDimension] - mean, 2);
			}
		}
		variance = totalSum / (totalCount - 1);

		return new StatisticsBundle(mean, variance);
	}

	private void densityCheck() {
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
			//TODO: Notify callback to build check contrast
		}

	}

	//TODO:
	private double calculateKNNDistance(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

}
