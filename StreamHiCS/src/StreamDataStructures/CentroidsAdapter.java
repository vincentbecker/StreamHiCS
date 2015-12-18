package streamdatastructures;

import centroids.FadingCentroids;
import centroids.AdaptingCentroid;
import centroids.Centroid;
import weka.core.Instance;

public class CentroidsAdapter extends SummarisationAdapter {

	/**
	 * The {@link CentroidsContainer} holding the {@link AdaptingCentroid}s.
	 */
	private FadingCentroids centroidsImplementation;
	
	public CentroidsAdapter(int horizon, double radius, double learningRate) {
		centroidsImplementation = new FadingCentroids();
		centroidsImplementation.horizonOption.setValue(horizon);
		centroidsImplementation.radiusOption.setValue(radius);
		centroidsImplementation.learningRateOption.setValue(learningRate);
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
	
	public Centroid[] getCentroids(){
		return centroidsImplementation.getCentroids();
	}

}
