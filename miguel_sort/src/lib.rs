use pyo3::prelude::*;
use::pyo3::wrap_pyfunction;

/// Sorting algorithm but worse xd
#[pyfunction]
fn miguel_sort(arr: &[u32]) -> Vec<u32> {
    let mut st  = vec![fasle; 2usize.pow(32)];
    arr.into_iter().for_each(|&f| {
        set[f as usize] = true;
    });
    set.into_iter()
        .enumerate()
        .filter(|&(_, b)| b)
        .map(|(i, _)| i as u32)
        .collect()
}

/// A Python module implemented in Rust.
#[pymodule]
fn miguel_sort(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(miguel_sort, m)?)?;
    Ok(())
}
