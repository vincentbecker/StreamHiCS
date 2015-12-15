package streamdatastructures;

import coreset.BucketManager;
import coreset.Point;
import moa.clusterers.streamkm.MTRandom;
import weka.core.Instance;

public class CoresetAdapter extends SummarisationAdapter {

	private BucketManager bucketManager;
	private boolean initialized = false;
	private int width = 0;
	private int coresetSize = 0;
	private int numberInstances = 0;
	private int d = 0;
	private Point[] currentCoreset;

	public CoresetAdapter(int width, int coresetSize) {
		this.width = width;
		this.coresetSize = coresetSize;
	}

	@Override
	public void addImpl(Instance instance) {
		if (!initialized) {
			d = instance.numAttributes();
			bucketManager = new BucketManager(this.width, instance.numAttributes(), this.coresetSize, new MTRandom(1));
			initialized = true;
		}

		bucketManager.insertPoint(new Point(instance, this.numberInstances));
		numberInstances++;
		currentCoreset = null;
	}

	@Override
	public void clearImpl() {
		initialized = false;
		currentCoreset = null;
		numberInstances = 0;
	}

	@Override
	public DataBundle[] getData() {
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(d);
		}

		int n = currentCoreset.length;
		if (n > 0) {
			double[][] points = new double[n][];
			double[] weights = new double[n];
			Point p;
			for (int i = 0; i < n; i++) {
				p = currentCoreset[i];
				points[i] = p.getCoordinates();
				weights[i] = p.getWeight();
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
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(d);
		}
		if (currentCoreset == null) {
			return 0;
		}

		return currentCoreset.length;
	}

}
