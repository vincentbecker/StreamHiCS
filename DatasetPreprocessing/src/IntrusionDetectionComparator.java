import java.util.Comparator;

public class IntrusionDetectionComparator implements Comparator<String> {

	@Override
	public int compare(String o1, String o2) {

		//Sorts normal vs not normal
		if (o1.endsWith("normal") && o2.endsWith("normal")) {
			return 0;
		} else if (o1.endsWith("normal") && !o2.endsWith("normal")) {
			return -1;
		} else if (!o1.endsWith("normal") && o2.endsWith("normal")) {
			return 1;
		}
		return 0;
	}

}
