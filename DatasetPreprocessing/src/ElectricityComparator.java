import java.util.Comparator;

public class ElectricityComparator implements Comparator<String> {

	@Override
	public int compare(String o1, String o2) {
		// Get last character. The class label is either UP or DOWN. Hence the
		// last character can be used to distinguish.
		char class1 = o1.charAt(o1.length() - 1);
		char class2 = o2.charAt(o2.length() - 1);

		// DOWN < UP
		if (class1 == 'N' && class2 == 'N') {
			return 0;
		} else if (class1 == 'P' && class2 == 'P') {
			return 0;
		} else if (class1 == 'N' && class2 == 'P') {
			return -1;
		} else if (class1 == 'P' && class2 == 'N') {
			return 1;
		}
		
		return -2;
	}
}