package streamDataStructures;

import java.util.Comparator;

public class ArrayIndexComparator implements Comparator<Integer> {
	private double[] array;

	public void setArray(double[] array) {
		this.array = array;
	}

	public Integer[] createIndexArray() {
		Integer[] indexes = new Integer[array.length];
		for (int i = 0; i < array.length; i++) {
			indexes[i] = i;
		}
		return indexes;
	}

	@Override
	public int compare(Integer index1, Integer index2) {
		if (array[index1] < array[index2]) {
			return -1;
		} else if (array[index1] == array[index2]) {
			return 0;
		} else {
			return 1;
		}
	}
}