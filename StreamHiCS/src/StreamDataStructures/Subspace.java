package StreamDataStructures;

import java.util.ArrayList;

/**
 * This class represents a subspace consisting of the dimensions it contains.
 * 
 * @author Vincent
 *
 */
public class Subspace {

	/**
	 * The dimensions the {@link Subspace} consists of.
	 */
	private ArrayList<Integer> dimensions;

	/**
	 * Adds the specified dimension to the {@link Subspace}. If the dimension is
	 * already contained in the {@link Subspace} nothing is changed.
	 * 
	 * @param dimension
	 *            The dimension to be added.
	 */
	public void addDimension(int dimension) {
		dimensions.add(dimension);
	}

	/**
	 * Returns the number of dimensions in the {@link Subspace}.
	 * 
	 * @return The number of dimensions in the {@link Subspace}.
	 */
	public int getSize() {
		return dimensions.size();
	}

	/**
	 * Returns the dimension at the specified index.
	 * 
	 * @param index
	 *            The index.
	 * @return The dimension at the specified index.
	 */
	public int getDimension(int index) {
		return (int) dimensions.get(index);
	}

	/**
	 * The dimensions of the {@link Subspace} as an int array.
	 * 
	 * @return An int array containing the dimensions.
	 */
	public int[] getDimensions() {
		int numberOfDimensions = dimensions.size();
		Integer[] temp = new Integer[numberOfDimensions];
		temp = dimensions.toArray(temp);
		int[] result = new int[numberOfDimensions];
		// Cast all elements to int
		for (int i = 0; i < numberOfDimensions; i++) {
			result[i] = (int) temp[i];
		}
		return result;
	}
	
	public void discardDimension(int index){
		dimensions.remove(index);
	}
	
	public Subspace copy(){
		Subspace c = new Subspace();
		for(int i = 0; i < this.getSize(); i++){
			c.dimensions.add(this.dimensions.get(i));
		}
		return c;
	}
	
	public boolean equals(Subspace s2){
		if(this.getSize() != s2.getSize()){
			return false;
		}
		for(int dimension : dimensions){
			if(!s2.dimensions.contains(dimension)){
				return false;
			}
		}
		return true;
	}
}
