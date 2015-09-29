package streamDataStructures;

import java.util.ArrayList;

import weka.clusterers.SelfOrganizingMap;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;

/**
 * This class represents a self organising map that collects the incoming
 * instances and is trained on them. It acts as a summarisation of the instance
 * information.
 * 
 * @author Vincent
 *
 */
public class SelfOrganizingMapContainer extends DataStreamContainer {

	/**
	 * The internal self organising map.
	 */
	private SelfOrganizingMap som;
	/**
	 * A list to hold all the new data until the next update of the som.
	 */
	private Instances trainingData;

	/**
	 * Creates a {@link SelfOrganizingMapContainer} object.
	 */
	public SelfOrganizingMapContainer(ArrayList<Attribute> attributes) {
		som = new SelfOrganizingMap();
		trainingData = new Instances("Training", attributes, 1000);
	}

	@Override
	public void add(Instance instance) {
		trainingData.add(instance);
	}

	@Override
	public int getNumberOfInstances() {
		// Number of neurons
		Instances inst = null;
		try {
			inst = som.getClusters();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return inst.size();
	}

	@Override
	public double[] getProjectedData(int dimension) {
		// Update som
		trainSOM();
		return null;
	}

	@Override
	public double[] getSlicedData(Subspace subspace, int dimension,
			double selectionAlpha) {
		// Update som
		trainSOM();
		return null;
	}

	/**
	 * Train the som, if any training data is available.
	 */
	public void trainSOM() {
		if (trainingData.size() > 0) {
			// Train SOM
			try {
				som.buildClusterer(trainingData);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			// Clear training data
			trainingData.clear();
			
			try {
				System.out.println(som.getClusters().toString());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

}
