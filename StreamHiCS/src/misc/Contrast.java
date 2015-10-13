package misc;

import streamDataStructures.Subspace;
import weka.core.Instance;

/**
 * This is an interface for the contrast calculation.
 * 
 * @author Vincent
 *
 */
public interface Contrast {

	/**
	 * Add an @link{Instance}.
	 * 
	 * @param instance
	 *            The @link{Instance} to be added.
	 */
	public void add(Instance instance);
	
	/**
	 * Clears all stored {@link Instance}s.
	 */
	public void clear();

	/**
	 * Calculates the contrast of the given @link{Subspace}.
	 * 
	 * @param subspace
	 *            The @link{Subspace} the contrast is calculated of.
	 * @return The contrast of the given @link{Subspace}.
	 */
	public double evaluateSubspaceContrast(Subspace subspace);

}
