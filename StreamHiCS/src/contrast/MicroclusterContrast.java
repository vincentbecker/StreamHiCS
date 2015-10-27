package contrast;

import changechecker.ChangeChecker;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import weka.core.Instance;

public class MicroclusterContrast extends Contrast {

	private AbstractClusterer microclusterImplementation;
	private Clustering microclusters;
	private ChangeChecker changeChecker;

	public MicroclusterContrast(Callback callback, int m, double alpha, AbstractClusterer microclusterImplementation, ChangeChecker changeChecker) {
		super(callback, m, alpha);
		this.microclusterImplementation = microclusterImplementation;
		this.changeChecker = changeChecker;
	}

	@Override
	public void add(Instance instance) {
		microclusterImplementation.trainOnInstanceImpl(instance);
		microclusters = null;
		if(changeChecker.poll() && changeChecker.checkForChange()){
			onAlarm();
		}
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
	public double[] getProjectedData(int referenceDimension) {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}

		int l = microclusters.size();
		double[] data = new double[l];
		for (int i = 0; i < l; i++) {
			data[i] = microclusters.get(i).getCenter()[referenceDimension];
		}

		return data;
	}

	@Override
	public double[] getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		if(microclusters.size() == 0){
			return new double[0];
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

		// Get the selected data from the last dimension and apply weights
		weights = getSelectedWeights(selectedIndexes);
		int totalWeight = 0;
		for (int i = 0; i < weights.length; i++) {
			weights[i] = Math.round(weights[i]);
			totalWeight += weights[i];
		}
		double[] slicedData = new double[totalWeight];
		int index = 0;
		double c;
		int referenceDimension = shuffledDimensions[shuffledDimensions.length - 1];
		for (int i = 0; i < selectedIndexes.size(); i++) {
			c = microclusters.get(selectedIndexes.getIndex(i)).getCenter()[referenceDimension];
			for (int j = 0; j < weights[i]; j++) {
				slicedData[index] = c;
				index++;
			}
		}
		return slicedData;
	}
	
	public Selection getSliceIndexes(int[] shuffledDimensions, double selectionAlpha) {
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		if(microclusters.size() == 0){
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


	private double[] getSelectedData(int dimension, Selection selectedIndexes) {
		int l = selectedIndexes.size();
		double[] data = new double[l];
		for (int i = 0; i < l; i++) {
			data[i] = microclusters.get(selectedIndexes.getIndex(i)).getCenter()[dimension];
		}

		return data;
	}

	private double[] getSelectedWeights(Selection selectedIndexes) {
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
		for(int i = 0; i < microclusters.size(); i++){
			points[i] = microclusters.get(i).getCenter();
		}
		return points;
	}
	
	public Clustering getMicroclusters(){
		if (microclusters == null) {
			microclusters = microclusterImplementation.getMicroClusteringResult();
		}
		return microclusters;
	}
}
