package changechecker;

import subspace.Subspace;
import weka.core.Instance;

/**
 * A default checking mechanism which returns true every time it is called and
 * thereby induces a new evaluation of the correlated subspaces.
 * 
 * @author Vincent
 *
 */
public class TimeCountChecker extends ChangeChecker {
	
	/**
	 * @param checkInterval The number of {@link Instance} that are observed before the
	 * {@link Subspace} contrasts are checked again.
	 */
	public TimeCountChecker(int checkInterval) {
		super(checkInterval);
	}

	@Override
	public boolean checkForChange() {
		return true;
	}

}
