import java.util.Comparator;

public class IntrusionDetectionComparator implements Comparator<String> {

	@Override
	public int compare(String o1, String o2) {

		String[] o1Split = o1.split(",");
		String class1 = o1Split[o1Split.length - 1];
		String[] o2Split = o2.split(",");
		String class2 = o2Split[o2Split.length - 1];

		// Sorts normal vs not normal. Not normal in lexicographic order
		/*
		 * if (class1.equals("normal") && class2.equals("normal")) { return 0; }
		 * else if (class1.equals("normal") && !class2.equals("normal")) {
		 * return -1; } else if (!class1.equals("normal") &&
		 * class2.equals("normal")) { return 1; } else { return
		 * class1.compareTo(class2); }
		 */
		
		return class1.compareTo(class2);
	}

}
