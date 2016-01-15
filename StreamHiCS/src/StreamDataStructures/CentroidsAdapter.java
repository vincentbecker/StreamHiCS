package streamdatastructures;

import centroids.FadingCentroids;
import centroids.Centroid;
import weka.core.Instance;

/**
 * This class represents an adapter to access data from a {@link Centroid}s
 * implementation.
 * 
 * @author Vincent
 *
 */
public class CentroidsAdapter extends SummarisationAdapter {

	/**
	 * The {@link FadingCentroids} instance holding the {@link Centroid}s.
	 */
	private FadingCentroids centroidsImplementation;

	/**
	 * Creates an instance of this class. The horizon is the amount of time a
	 * single {@link Centroid} would "survive" (not being faded away) without
	 * any reinforcement through added points. The learning rate determines how
	 * fast a {@link Centroid} adapts to a new point in its environment (depends
	 * on the {@link Centroid} implementation).
	 * 
	 * @param horizon
	 *            The horizon
	 * @param radius
	 *            The radius of a {@link Centroid}
	 * @param learningRate
	 *            The learning rate
	 */
	public CentroidsAdapter(int horizon, double radius, double learningRate, String version) {
		centroidsImplementation = new FadingCentroids();
		centroidsImplementation.horizonOption.setValue(horizon);
		centroidsImplementation.radiusOption.setValue(radius);
		centroidsImplementation.learningRateOption.setValue(learningRate);
		centroidsImplementation.centroidVersionOption.setValue(version);
		centroidsImplementation.prepareForUse();
	}

	@Override
	public void addImpl(Instance instance) {
		centroidsImplementation.add(instance);
	}

	@Override
	public void clearImpl() {
		centroidsImplementation.clear();
	}

	@Override
	public DataBundle[] getData() {
		Centroid[] centroids = centroidsImplementation.getCentroids();

		int n = centroids.length;
		if (n > 0) {
			int d = centroids[0].getCentre().length;
			double[][] points = new double[n][];
			double[] weights = new double[n];
			Centroid c;
			for (int i = 0; i < n; i++) {
				c = centroids[i];
				points[i] = c.getCentre();
				weights[i] = c.getWeight(-1);
			}

			// Construct the DataBundles from the data
			DataBundle[] data = new DataBundle[d];

			for (int dim = 0; dim < d; dim++) {
				double[] dimData = new double[n];
				double[] w = new double[n];
				for (int i = 0; i < n; i++) {
					dimData[i] = points[i][dim];
					// Copying weights, necessary since manipulated later
					w[i] = weights[i];
				}

				data[dim] = new DataBundle(dimData, w);
			}

			return data;
		} else {
			return null;
		}
	}

	@Override
	public int getNumberOfElements() {
		return centroidsImplementation.getNumberOfInstances();
	}

	/**
	 * Returns the array of {@link Centroid}s.
	 * 
	 * @return The {@link Centroid}s array.
	 */
	public Centroid[] getCentroids() {
		return centroidsImplementation.getCentroids();
	}

	/**
	 * Returns the number of faded {@link Centroid}s throughout the streaming
	 * process.
	 * 
	 * @return The number of faded {@link Centroid}s throughout the streaming
	 *         process.
	 */
	public int getFadedCount() {
		return centroidsImplementation.faded;
	}

}
