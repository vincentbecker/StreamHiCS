package contrast;

import centroids.AdaptingCentroids;
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

	public CentroidContrast(Callback callback, int numberOfDimensions, int m, double alpha, int checkCount) {
		super(callback, m, alpha);
		centroids = new AdaptingCentroids(this, numberOfDimensions, 0.01, 0.01, 500, checkCount, 0.1, 0.8);
	}

	@Override
	public void add(Instance instance) {
		centroids.add(instance);
	}

	@Override
	public void clear() {
		centroids.clear();
	}

	@Override
	public double[] getProjectedData(int referenceDimension) {
		return centroids.getProjectedData(referenceDimension);
	}

	@Override
	public double[] getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		return centroids.getSlicedData(shuffledDimensions, selectionAlpha);
	}
}
