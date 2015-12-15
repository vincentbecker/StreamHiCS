package streamdatastructures;

import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import weka.core.Instance;

public class MicroclusterAdapter extends SummarisationAdapter {

	private AbstractClusterer microclusterImplementation;
	private Clustering microclusters;

	public MicroclusterAdapter(AbstractClusterer microclusterImplementation) {
		this.microclusterImplementation = microclusterImplementation;
	}

	@Override
	public void addImpl(Instance instance) {
		microclusterImplementation.trainOnInstance(instance);
		microclusters = null;
	}

	@Override
	public void clearImpl() {
		microclusterImplementation.resetLearning();
		microclusters = null;
	}

	@Override
	public DataBundle[] getData() {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}

		int n = microclusters.size();
		if (n > 0) {
			int d = microclusters.get(0).getCenter().length;
			double[][] points = new double[n][];
			double[] weights = new double[n];
			Cluster c;
			for (int i = 0; i < n; i++) {
				c = microclusters.get(i);
				points[i] = c.getCenter();
				weights[i] = c.getWeight();
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
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		if (microclusters == null) {
			return 0;
		}
		return microclusters.size();
	}

}
