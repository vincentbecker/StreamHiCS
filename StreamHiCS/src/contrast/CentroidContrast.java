package contrast;

import centroids.AdaptingCentroids;
import centroids.Centroid;
import centroids.CentroidsContainer;
import weka.core.Instance;

/**
 * This class represents a contrast calculation based on centroids (one could
 * also refer to micro-clusters).
 * 
 * @author Vincent
 *
 */
public class CentroidContrast extends Contrast {

	/**
	 * The {@link CentroidsContainer} holding the {@link Centroids}.
	 */
	private CentroidsContainer centroids;
	private int numberOfDimensions;

	public CentroidContrast(int numberOfDimensions, int m, double alpha, double fadingLambda,
			double radius, double weightThreshold, double learningRate) {
		super(m, alpha);
		centroids = new AdaptingCentroids(numberOfDimensions, fadingLambda, radius, weightThreshold, learningRate);
		this.numberOfDimensions = numberOfDimensions;
	}

	@Override
	public void add(Instance instance) {
		centroids.add(instance);
	}

	@Override
	public void clear() {
		centroids.clear();
	}

	public Centroid[] getCentroids() {
		return centroids.getCentroids();
	}

	@Override
	public DataBundle getProjectedData(int referenceDimension) {
		return centroids.getProjectedData(referenceDimension);
	}

	@Override
	public DataBundle getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		return centroids.getSlicedData(shuffledDimensions, selectionAlpha);
	}

	public Selection getSliceIndexes(int[] shuffledDimensions, double selectionAlpha) {
		return ((AdaptingCentroids) centroids).getSliceIndexes(shuffledDimensions, selectionAlpha);
	}

	@Override
	public int getNumberOfElements() {
		return centroids.getNumberOfInstances();
	}

	@Override
	public double[][] getUnderlyingPoints() {
		Centroid[] cs = centroids.getCentroids();
		double[][] points = new double[cs.length][numberOfDimensions];
		for(int i = 0; i < cs.length; i++){
			points[i] = cs[i].getVector();
		}
		return points;
	}
}
