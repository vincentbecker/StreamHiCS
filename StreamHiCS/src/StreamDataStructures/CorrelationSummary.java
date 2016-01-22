package streamdatastructures;

import weka.core.Instance;

/**
 * Maintains a stream summary in order to calculate the Pearson's correlation
 * coefficient for each pair of dimensions.
 * 
 * @author Vincent
 *
 */
public class CorrelationSummary {

	/**
	 * The number of dimensions of the stream.
	 */
	private int numberOfDimensions;

	/**
	 * The number of instances which have arrived.
	 */
	private double n;

	/**
	 * The linear sum of the instances values for each dimension.
	 */
	private double[] linearSums;

	/**
	 * The squared sum of the instances values for each dimension.
	 */
	private double[] squaredSums;

	/**
	 * The sums of the products for each pair of dimensions.
	 */
	private double[][] products;

	private double fadingFactor;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the stream
	 */
	public CorrelationSummary(int numberOfDimensions, int horizon) {
		this.numberOfDimensions = numberOfDimensions;
		linearSums = new double[numberOfDimensions];
		squaredSums = new double[numberOfDimensions];
		products = new double[numberOfDimensions][numberOfDimensions];
		double negLambda = Math.log(0.01) / Math.log(2) / (double) horizon;
		fadingFactor = Math.pow(2, negLambda);
	}

	/**
	 * Adds a stream instance's values to the summary. Before we fade the
	 * summary.
	 * 
	 * @param instance
	 *            The instance
	 */
	public void addInstance(Instance instance) {
		double[] vector = instance.toDoubleArray();
		for (int i = 0; i < numberOfDimensions; i++) {
			linearSums[i] *= fadingFactor;
			linearSums[i] += vector[i];
			squaredSums[i] *= fadingFactor;
			squaredSums[i] += vector[i] * vector[i];
			for (int j = i + 1; j < numberOfDimensions; j++) {
				products[i][j] *= fadingFactor;
				products[i][j] += vector[i] * vector[j];
			}
		}
		n *= fadingFactor;
		n++;
	}

	/**
	 * Calculates the Pearson's correlation coefficient of two given dimensions
	 * form the summary.
	 * 
	 * @param dim1
	 *            The first dimension
	 * @param dim2
	 *            The second dimension
	 * @return The Pearson's correlation coefficient for the two dimensions.
	 */
	public double calculateCorrelation(int dim1, int dim2) {
		assert (dim1 != dim2);
		if (dim1 > dim2) {
			int temp = dim1;
			dim1 = dim2;
			dim2 = temp;
		}
		double a = linearSums[dim1];
		double a2 = squaredSums[dim1];
		double b = linearSums[dim2];
		double b2 = squaredSums[dim2];
		return (products[dim1][dim2] - a * b / n) / (Math.sqrt(a2 - a * a / n) * Math.sqrt(b2 - b * b / n));
	}

	/**
	 * Returns a matrix containing the correlation coefficients.
	 * 
	 * @return A matrix containing the correlation coefficients.
	 */
	public double[][] getCorrelationMatrix() {
		double[][] coefficientMatrix = new double[numberOfDimensions][numberOfDimensions];
		double coefficient;
		for (int i = 0; i < numberOfDimensions; i++) {
			for (int j = i + 1; j < numberOfDimensions; j++) {
				coefficient = calculateCorrelation(i, j);
				coefficientMatrix[i][j] = coefficient;
				coefficientMatrix[j][i] = coefficient;
			}
		}
		return coefficientMatrix;
	}

	/**
	 * Clears the correlation summary by resetting all values to 0.
	 */
	public void clear() {
		for (int i = 0; i < numberOfDimensions; i++) {
			linearSums[i] = 0;
			squaredSums[i] = 0;
			for (int j = i + 1; j < numberOfDimensions; j++) {
				products[i][j] = 0;
			}
		}
		n = 0;
	}
}
