package contrast;

import centroids.AdaptingCentroids;
import centroids.Centroid;
import centroids.CentroidsContainer;
import centroids.ChangeChecker;
import streamDataStructures.Selection;
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

	public CentroidContrast(Callback callback, int numberOfDimensions, int m, double alpha, double fadingLambda,
			double radius, int checkCount, double weightThreshold, double learningRate, ChangeChecker changeChecker) {
		super(callback, m, alpha);
		centroids = new AdaptingCentroids(this, numberOfDimensions, fadingLambda, radius, checkCount, weightThreshold, learningRate, changeChecker);
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
	public double[] getProjectedData(int referenceDimension) {
		return centroids.getProjectedData(referenceDimension);
	}

	@Override
	public double[] getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		return centroids.getSlicedData(shuffledDimensions, selectionAlpha);
	}

	public Selection getSliceIndexes(int[] shuffledDimensions, double selectionAlpha) {
		return ((AdaptingCentroids) centroids).getSliceIndexes(shuffledDimensions, selectionAlpha);
	}

	@Override
	public int getNumberOfElements() {
		return centroids.getNumberOfInstances();
	}
}
