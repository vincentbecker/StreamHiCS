package streamdatastructures;

import centroids.AdaptingCentroids;
import centroids.Centroid;
import centroids.CentroidsContainer;

import weka.core.Instance;

public class CentroidsAdapter extends SummarisationAdapter {

	/**
	 * The {@link CentroidsContainer} holding the {@link Centroid}s.
	 */
	private CentroidsContainer centroidsContainer;
	private Centroid[] centroids;
	
	public CentroidsAdapter(int numberOfDimensions, double fadingLambda, double radius, double weightThreshold, double learningRate) {
		centroidsContainer = new AdaptingCentroids(numberOfDimensions, fadingLambda, radius, weightThreshold, learningRate);
	}
	
	@Override
	public void addImpl(Instance instance) {
		centroidsContainer.add(instance);
		centroids = null;
	}

	@Override
	public void clearImpl() {
		centroidsContainer.clear();
		centroids = null;
	}

	@Override
	public DataBundle[] getData() {
		if (centroids == null) {
			centroids = centroidsContainer.getCentroids();
		}
		
		int n = centroids.length;
		if (n > 0) {
			int d = centroids[0].getVector().length;
			double[][] points = new double[n][];
			double[] weights = new double[n];
			Centroid c;
			for (int i = 0; i < n; i++) {
				c = centroids[i];
				points[i] = c.getVector();
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
		return centroidsContainer.getNumberOfInstances();
	}
	
	public Centroid[] getCentroids(){
		return centroidsContainer.getCentroids();
	}

}
