use pyo3::prelude::*;
use::pyo3::wrap_pyfunction;

/// Sorting algorithm but worse xd
#[pyfunction]
fn miguel_sort(arr: Vec<u32>) -> Vec<u32> {
    let mut st  = vec![false; 2usize.pow(32)];
    for &f in & arr {
        set[f as usize] = true;
    }

    set.into_iter()
        .enumerate()
        .filter(|&(_, b)| b)
        .map(|(i, _)| i as u32)
        .collect()
}

/// A Python module implemented in Rust.
#[pymodule]
fn miguel_sort_module(m: &PyModule<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(miguel_sort, m)?)?;
    Ok(())
}
