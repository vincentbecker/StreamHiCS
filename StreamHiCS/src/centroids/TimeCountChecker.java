package centroids;

import java.util.ArrayList;

/**
 * A default checking mechanism which returns true every time it is called and
 * thereby induces a new evaluation of the correlated subspaces.
 * 
 * @author Vincent
 *
 */
public class TimeCountChecker extends ChangeChecker {

	@Override
	public boolean checkForChange(ArrayList<Centroid> centroids) {
		return true;
	}

}
