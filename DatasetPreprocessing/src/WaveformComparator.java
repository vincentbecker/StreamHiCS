import java.util.Comparator;

public class WaveformComparator implements Comparator<String> {

	@Override
	public int compare(String o1, String o2) {
		int class1 = Integer.parseInt("" + o1.charAt(o1.length() - 1));
		int class2 = Integer.parseInt("" + o2.charAt(o2.length() - 1));

		if (class1 < class2) {
			return -1;
		} else if (class1 > class2) {
			return 1;
		} else {
			return 0;
		}
	}
}
