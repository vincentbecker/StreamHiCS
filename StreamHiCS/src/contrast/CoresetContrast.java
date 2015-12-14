package contrast;

import coreset.BucketManager;
import coreset.Point;
import moa.clusterers.streamkm.MTRandom;
import weka.core.Instance;

public class CoresetContrast extends Contrast {

	private BucketManager bucketManager;
	private boolean initialized = false;
	private int width = 0;
	private int coresetSize = 0;
	private int numberInstances = 0;
	private int numberDimensions = 0;
	private Point[] currentCoreset;

	public CoresetContrast(int m, double alpha, int width, int coresetSize) {
		super(m, alpha);
		this.width = width;
		this.coresetSize = coresetSize;
	}

	@Override
	public void add(Instance instance) {
		if (!initialized) {
			numberDimensions = instance.numAttributes();
			bucketManager = new BucketManager(this.width, instance.numAttributes(), this.coresetSize, new MTRandom(1));
			initialized = true;
		}

		bucketManager.insertPoint(new Point(instance, this.numberInstances));
		numberInstances++;
		currentCoreset = null;
	}

	@Override
	public void clear() {
		initialized = false;
		currentCoreset = null;
		numberInstances = 0;
	}

	@Override
	public int getNumberOfElements() {
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(numberDimensions);
		}
		if (currentCoreset == null) {
			return 0;
		}

		return currentCoreset.length;
	}

	@Override
	public DataBundle getProjectedData(int referenceDimension) {
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(numberDimensions);
		}

		int l = currentCoreset.length;

		Point p;
		double[] data = new double[l];
		double[] weights = new double[l];
		for (int i = 0; i < l; i++) {
			p = currentCoreset[i];
			data[i] = p.getCoordinates()[referenceDimension];
			weights[i] = p.getWeight();
		}

		return new DataBundle(data, weights);
	}

	@Override
	public DataBundle getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(numberDimensions);
		}
		if (currentCoreset.length == 0) {
			return new DataBundle(new double[0], new double[0]);
		}

		double[] dimData;
		double[] weights;
		Selection selectedIndexes = new Selection(currentCoreset.length, selectionAlpha);
		// Fill the list with all the indexes
		selectedIndexes.fillRange();

		for (int i = 0; i < shuffledDimensions.length - 1; i++) {
			// Get all the data for the specific dimension that is selected
			dimData = getSelectedData(shuffledDimensions[i], selectedIndexes);
			weights = getSelectedWeights(selectedIndexes);
			// Reduce the number of indexes according to a new selection in
			// the current dimension
			selectedIndexes.selectWithWeights(dimData, weights);
		}

		// Get the selected data from the last dimension and apply weights
		dimData = getSelectedData(shuffledDimensions[shuffledDimensions.length - 1], selectedIndexes);
		weights = getSelectedWeights(selectedIndexes);

		return new DataBundle(dimData, weights);
	}

	public Selection getSliceIndexes(int[] shuffledDimensions, double selectionAlpha) {
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(numberDimensions);
		}
		if (currentCoreset.length == 0) {
			return new Selection(0, selectionAlpha);
		}

		double[] dimData;
		double weights[];
		Selection selectedIndexes = new Selection(currentCoreset.length, selectionAlpha);
		// Fill the list with all the indexes
		selectedIndexes.fillRange();

		for (int i = 0; i < shuffledDimensions.length - 1; i++) {
			// Get all the data for the specific dimension that is selected
			dimData = getSelectedData(shuffledDimensions[i], selectedIndexes);
			weights = getSelectedWeights(selectedIndexes);
			// Reduce the number of indexes according to a new selection in
			// the current dimension
			selectedIndexes.selectWithWeights(dimData, weights);
		}

		return selectedIndexes;
	}

	public double[] getSelectedData(int dimension, Selection selectedIndexes) {
		int l = selectedIndexes.size();
		double[] data = new double[l];
		for (int i = 0; i < l; i++) {
			data[i] = currentCoreset[selectedIndexes.getIndex(i)].getCoordinates()[dimension];
		}

		return data;
	}

	public double[] getSelectedWeights(Selection selectedIndexes) {
		int l = selectedIndexes.size();
		double[] weights = new double[l];
		for (int i = 0; i < l; i++) {
			weights[i] = currentCoreset[selectedIndexes.getIndex(i)].getWeight();
		}

		return weights;
	}

	@Override
	public double[][] getUnderlyingPoints() {
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(numberDimensions);
		}

		double[][] points = new double[currentCoreset.length][];
		for (int i = 0; i < currentCoreset.length; i++) {
			points[i] = currentCoreset[i].getCoordinates();
		}
		return points;
	}
}
