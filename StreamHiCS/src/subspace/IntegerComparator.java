package subspace;

import java.util.Comparator;

/**
 * This class represents a comparator for the type {@link Integer}.
 * 
 * @author Vincent
 *
 */
public class IntegerComparator implements Comparator<Integer> {

	@Override
	public int compare(Integer i1, Integer i2) {
		if (i1 < i2) {
			return -1;
		} else if (i1 > i2) {
			return 1;
		} else {
			return 0;
		}
	}
}
