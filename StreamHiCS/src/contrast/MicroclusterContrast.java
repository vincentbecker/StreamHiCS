package contrast;

import changechecker.ChangeChecker;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import weka.core.Instance;

public class MicroclusterContrast extends Contrast {

	private AbstractClusterer microclusterImplementation;
	private Clustering microclusters;
	

	public MicroclusterContrast(Callback callback, int m, double alpha, AbstractClusterer microclusterImplementation,
			ChangeChecker changeChecker) {
		super(callback, m, alpha, changeChecker);
		this.microclusterImplementation = microclusterImplementation;
	}

	@Override
	public void addImpl(Instance instance) {
		microclusterImplementation.trainOnInstanceImpl(instance);
		microclusters = null;
	}

	@Override
	public void clear() {
		microclusterImplementation.resetLearningImpl();
		microclusters = null;
	}

	@Override
	public int getNumberOfElements() {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		return microclusters.size();
	}

	@Override
	public DataBundle getProjectedData(int referenceDimension) {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		
		int l = microclusters.size();
		
		double[] data = new double[l];
		for (int i = 0; i < l; i++) {
			data[i] = microclusters.get(i).getCenter()[referenceDimension];
		}
		
		double[] weights = new double[l];
		for (int i = 0; i < l; i++) {
			weights[i] = microclusters.get(i).getWeight();
		}
		
		return new DataBundle(data, weights);
	}

	@Override
	public DataBundle getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		if (microclusters.size() == 0) {
			return new DataBundle(new double[0], new double[0]);
		}

		double[] dimData;
		double[] weights;
		Selection selectedIndexes = new Selection(microclusters.size(), selectionAlpha);
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
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		if (microclusters.size() == 0) {
			return new Selection(0, selectionAlpha);
		}

		double[] dimData;
		double weights[];
		Selection selectedIndexes = new Selection(microclusters.size(), selectionAlpha);
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
			data[i] = microclusters.get(selectedIndexes.getIndex(i)).getCenter()[dimension];
		}

		return data;
	}

	public double[] getSelectedWeights(Selection selectedIndexes) {
		int l = selectedIndexes.size();
		double[] weights = new double[l];
		for (int i = 0; i < l; i++) {
			weights[i] = microclusters.get(selectedIndexes.getIndex(i)).getWeight();
		}

		return weights;
	}

	@Override
	public double[][] getUnderlyingPoints() {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}

		double[][] points = new double[microclusters.size()][];
		for (int i = 0; i < microclusters.size(); i++) {
			points[i] = microclusters.get(i).getCenter();
		}
		return points;
	}

	public Clustering getMicroclusters() {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		return microclusters;
	}
}
