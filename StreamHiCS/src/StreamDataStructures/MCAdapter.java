package streamdatastructures;

import microclusters.Microcluster;
import microclusters.FadingMicroclusters;
import weka.core.Instance;

/**
 * This class represents an adapter to access data from a {@link Microcluster}s
 * implementation.
 * 
 * @author Vincent
 *
 */
public class MCAdapter extends SummarisationAdapter {

	/**
	 * The {@link FadingMicroclusters} instance holding the {@link Microcluster}s.
	 */
	private FadingMicroclusters microclusterImplementation;

	/**
	 * Creates an instance of this class. The horizon is the amount of time a
	 * single {@link Microcluster} would "survive" (not being faded away) without
	 * any reinforcement through added points. The learning rate determines how
	 * fast a {@link Microcluster} adapts to a new point in its environment (depends
	 * on the {@link Microcluster} implementation).
	 * 
	 * @param horizon
	 *            The horizon
	 * @param radius
	 *            The radius of a {@link Microcluster}
	 * @param learningRate
	 *            The learning rate
	 */
	public MCAdapter(int horizon, double radius, double learningRate, String version) {
		microclusterImplementation = new FadingMicroclusters();
		microclusterImplementation.horizonOption.setValue(horizon);
		microclusterImplementation.radiusOption.setValue(radius);
		microclusterImplementation.learningRateOption.setValue(learningRate);
		microclusterImplementation.microclusterVersionOption.setValue(version);
		microclusterImplementation.prepareForUse();
	}

	@Override
	public void addImpl(Instance instance) {
		microclusterImplementation.trainOnInstance(instance);
	}

	@Override
	public void clearImpl() {
		microclusterImplementation.resetLearning();
	}

	@Override
	public DataBundle[] getData() {
		Microcluster[] centroids = microclusterImplementation.getMicroclusters();

		int n = centroids.length;
		if (n > 0) {
			int d = centroids[0].getCentre().length;
			double[][] points = new double[n][];
			double[] weights = new double[n];
			Microcluster c;
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
		return microclusterImplementation.getNumberOfInstances();
	}

	/**
	 * Returns the array of {@link Microcluster}s.
	 * 
	 * @return The {@link Microcluster}s array.
	 */
	public Microcluster[] getCentroids() {
		return microclusterImplementation.getMicroclusters();
	}

	/**
	 * Returns the number of faded {@link Microcluster}s throughout the streaming
	 * process.
	 * 
	 * @return The number of faded {@link Microcluster}s throughout the streaming
	 *         process.
	 */
	public int getFadedCount() {
		return microclusterImplementation.faded;
	}

}
