package changedetection;

import weka.core.Instance;

/**
 * This is an abstract class for streaming models running on subspaces.
 * 
 * @author Vincent
 *
 */
public interface SubspaceModel {

	/**
	 * Projects a given instance to the model's subspace.
	 * 
	 * @param instance
	 *            The instance
	 * @return A new instance, containing only the dimensions of the input
	 *         instance, which are contained in the model's subspace.
	 */
	public Instance projectInstance(Instance instance);

	/**
	 * Get the prediction for the given {@link Instance}. 
	 * 
	 * @param instance The instance
	 * @return The class prediction for te given {@link Instance}
	 */
	public int getClassPrediction(Instance instance);
	
	/**
	 * Get the current accuracy of the internal classifier.
	 * 
	 * @return the current accuracy of the internal classifier.
	 */
	public double getAccuracy();
}
