package streamDataStructures;

import kohonen.LearningDataFromStream;
import kohonen.WTMLearningFunction;
import learningFactorFunctional.HiperbolicFunctionalFactor;
import metrics.EuclidesMetric;
import network.DefaultNetwork;
import topology.GaussNeighbourhoodFunction;
import topology.MatrixTopology;
import weka.core.Instance;

/**
 * This class represents a self organising map that collects the incoming
 * instances and is trained on them. It acts as a summarisation of the instance
 * information.
 * 
 * @author Vincent
 *
 */
public class SelfOrganizingMap extends DataStreamContainer {

	/**
	 * The number of dimensions in the data space.
	 */
	private int numberOfDimensions;
	/**
	 * The data container to hold the data until the next update.
	 */
	private LearningDataFromStream data;
	/**
	 * The topology of the network.
	 */
	private MatrixTopology topology;
	/**
	 * The Kohonen network.
	 */
	private DefaultNetwork network;
	/**
	 * The learning function.
	 */
	private WTMLearningFunction learning;
	/**
	 * A flag showing if the network is up to date.
	 */
	private boolean updated = false;

	/**
	 * Creates a {@link SelfOrganizingMap} object.
	 * 
	 * @param edgeLength
	 *            The number of neurons on an egde. The topology is a square, so
	 *            the number of neurons will be the square.
	 */
	public SelfOrganizingMap(int numberOfDimensions, int edgeLength) {
		topology = new MatrixTopology(edgeLength, edgeLength);
		this.numberOfInstances = topology.getNumbersOfNeurons();
		double[] maxWeight = new double[numberOfDimensions];
		// Setting all maxWeight values to 1
		for (int i = 0; i < numberOfDimensions; i++) {
			maxWeight[i] = 1;
		}
		network = new DefaultNetwork(numberOfDimensions, maxWeight, topology);
		data = new LearningDataFromStream();
		HiperbolicFunctionalFactor learningFunction = new HiperbolicFunctionalFactor(1, 1);
		learning = new WTMLearningFunction(network, 100, new EuclidesMetric(), data, learningFunction,
				new GaussNeighbourhoodFunction(2));
	}

	@Override
	public void add(Instance instance) {
		double[] dataPoint = new double[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			dataPoint[i] = instance.value(i);
		}
		data.addData(dataPoint);
		updated = false;
	}

	@Override
	public void clear() {
		network.reset();
		data.clear();
	}

	@Override
	public int getNumberOfInstances() {
		return topology.getNumbersOfNeurons();
	}

	// Check if the network is up to date and then get the data.
	@Override
	public double[] getProjectedData(int dimension) {
		if (!updated) {
			trainSOM();
		}
		return super.getProjectedData(dimension);
	}

	// Check if the network is up to date and then get the data.
	@Override
	public double[] getSlicedData(int[] dimensions, double selectionAlpha) {
		if (!updated) {
			trainSOM();
		}
		return super.getSlicedData(dimensions, selectionAlpha);
	}

	/**
	 * Train the som.
	 */
	public void trainSOM() {
		learning.learn();
		// System.out.println(network);
		data.clear();
		updated = true;
	}

	public double[] getSelectedData(int dimension, Selection selectedIndexes) {
		double[] data = new double[selectedIndexes.size()];
		for (int i = 0; i < selectedIndexes.size(); i++) {
			data[i] = network.getNeuron(selectedIndexes.getIndex(i)).getWeight()[dimension];
		}
		return data;
	}
}
