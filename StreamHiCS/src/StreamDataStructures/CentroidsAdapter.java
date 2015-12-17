package streamdatastructures;

import centroids.AdaptingCentroids;
import centroids.Centroid;

import weka.core.Instance;

public class CentroidsAdapter extends SummarisationAdapter {

	/**
	 * The {@link CentroidsContainer} holding the {@link Centroid}s.
	 */
	private AdaptingCentroids centroidsImplementation;
	
	public CentroidsAdapter(double fadingLambda, double radius, double weightThreshold, double learningRate) {
		centroidsImplementation = new AdaptingCentroids(fadingLambda, radius, weightThreshold, learningRate);
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
		return centroidsImplementation.getNumberOfInstances();
	}
	
	public Centroid[] getCentroids(){
		return centroidsImplementation.getCentroids();
	}

}
