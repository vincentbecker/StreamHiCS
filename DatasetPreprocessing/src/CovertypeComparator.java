import java.util.Comparator;

public class CovertypeComparator implements Comparator<String> {

	@Override
	public int compare(String o1, String o2) {
		// Get last character. That is the class label (single-digit)
		int class1 = (int) o1.charAt(o1.length() - 1);
		int class2 = (int) o2.charAt(o2.length() - 1);

		if (class1 < class2) {
			return -1;
		} else if (class1 > class2) {
			return 1;
		} else {
			return 0;
		}
	}

}
