use calamine::{open_workbook, DataType, Reader, Xlsx};
use std::sync::Arc;
use arrow::array::{Array, ArrayRef, Float64Builder, Int32Builder, StringBuilder, BooleanBuilder};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Seek};
use arrow::csv::ReaderBuilder;

fn read_excel(file_path: &str, sheet_name: &str) -> Result<RecordBatch, Box<dyn Error>> {
    let mut workbook: Xlsx<_> = open_workbook(file_path)?;
    let range = workbook.worksheet_range(sheet_name).ok_or(format!("Cannot find sheet: {}", sheet_name))??;
    
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    if let Some(first_row) = range.rows().next() {
        for col_idx in 0..range.get_size().1 {
            let column_name = match &first_row[col_idx] {
                DataType::String(s) => s.clone(),
                _ => format!("column_{}", col_idx),
            };

            let mut int_builder = Int32Builder::new();
            let mut float_builder = Float64Builder::new();
            let mut str_builder = StringBuilder::new();
            let mut bool_builder = BooleanBuilder::new();

            let mut is_int = true;
            let mut is_float = true;
            let mut is_str = true;
            let is_bool = true;

            for (_row_idx, row) in range.rows().enumerate().skip(1) { // Skip the first row as it's used for column names
                match &row[col_idx] {
                    DataType::Int(v) => {
                        int_builder.append_value(*v as i32);
                        float_builder.append_value(*v as f64);
                        str_builder.append_value(&v.to_string());
                        bool_builder.append_value(*v != 0);
                    }
                    DataType::Float(v) => {
                        is_int = false;
                        float_builder.append_value(*v);
                        str_builder.append_value(&v.to_string());
                        bool_builder.append_value(*v != 0.0);
                    }
                    DataType::String(v) => {
                        is_int = false;
                        is_float = false;
                        str_builder.append_value(v);
                        bool_builder.append_value(v.to_lowercase() == "true" || v == "1");
                    }
                    DataType::Bool(v) => {
                        is_int = false;
                        is_float = false;
                        is_str = false;
                        bool_builder.append_value(*v);
                        str_builder.append_value(&v.to_string());
                    }
                    _ => {
                        int_builder.append_null();
                        float_builder.append_null();
                        str_builder.append_null();
                        bool_builder.append_null();
                    }
                }
            }

            let field: Field;
            let array: ArrayRef;

            if is_int {
                field = Field::new(&column_name, ArrowDataType::Int32, true);
                array = Arc::new(int_builder.finish());
            } else if is_float {
                field = Field::new(&column_name, ArrowDataType::Float64, true);
                array = Arc::new(float_builder.finish());
            } else if is_bool && !is_str {
                field = Field::new(&column_name, ArrowDataType::Boolean, true);
                array = Arc::new(bool_builder.finish());
            } else {
                field = Field::new(&column_name, ArrowDataType::Utf8, true);
                array = Arc::new(str_builder.finish());
            }

            fields.push(field);
            columns.push(array);
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns)?;

    Ok(batch)
}

fn read_csv(file_path: &str) -> Result<RecordBatch, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);

    // Infer the schema from the CSV file
    let (inferred_schema, _) = arrow::csv::reader::infer_reader_schema(&mut reader, b',', Some(1024), true)?;
    reader.rewind()?;

    // Create a CSV reader with the inferred schema
    let mut csv_reader = ReaderBuilder::new()
        .has_header(true)
        .with_schema(Arc::new(inferred_schema))
        .build(reader)?;

    let batch = csv_reader.next().transpose()?.unwrap();
    Ok(batch)
}

#[pyfunction]
fn read_excel_py(py: Python, file_path: String, sheet_name: String) -> PyResult<PyObject> {
    let batch = read_excel(&file_path, &sheet_name).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let arrow_py = PyModule::import(py, "pyarrow")?;

    // Convert RecordBatch to PyObject
    let schema = batch.schema();
    let schema_fields: Vec<PyObject> = schema
        .fields()
        .iter()
        .map(|f| arrow_py.call_method1("field", (f.name(), f.data_type().to_string())).unwrap().into())
        .collect();
    let schema_py = arrow_py.call_method1("schema", (PyList::new(py, &schema_fields),))?;

    let columns: Vec<PyObject> = batch
        .columns()
        .iter()
        .map(|array| {
            match array.data_type() {
                ArrowDataType::Int32 => {
                    let arr = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                    let values: Vec<PyObject> = (0..array.len()).map(|i| {
                        if arr.is_null(i) {
                            py.None()
                        } else {
                            arr.value(i).into_py(py)
                        }
                    }).collect();
                    arrow_py.call_method1("array", (PyList::new(py, &values),)).unwrap().into_py(py)
                }
                ArrowDataType::Float64 => {
                    let arr = array.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                    let values: Vec<PyObject> = (0..array.len()).map(|i| {
                        if arr.is_null(i) {
                            py.None()
                        } else {
                            arr.value(i).into_py(py)
                        }
                    }).collect();
                    arrow_py.call_method1("array", (PyList::new(py, &values),)).unwrap().into_py(py)
                }
                ArrowDataType::Utf8 => {
                    let arr = array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                    let values: Vec<PyObject> = (0..array.len()).map(|i| {
                        if arr.is_null(i) {
                            py.None()
                        } else {
                            arr.value(i).into_py(py)
                        }
                    }).collect();
                    arrow_py.call_method1("array", (PyList::new(py, &values),)).unwrap().into_py(py)
                }
                ArrowDataType::Boolean => {
                    let arr = array.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                    let values: Vec<PyObject> = (0..array.len()).map(|i| {
                        if arr.is_null(i) {
                            py.None()
                        } else {
                            arr.value(i).into_py(py)
                        }
                    }).collect();
                    arrow_py.call_method1("array", (PyList::new(py, &values),)).unwrap().into_py(py)
                }
                _ => panic!("Unsupported data type")
            }
        })
        .collect();

    let columns_py = PyList::new(py, &columns);
    let batch_py = arrow_py.call_method1("record_batch", (columns_py, schema_py))?;
    Ok(batch_py.into())
}

#[pyfunction]
fn read_csv_py(py: Python, file_path: String) -> PyResult<PyObject> {
    let batch = read_csv(&file_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let arrow_py = PyModule::import(py, "pyarrow")?;

    // Convert RecordBatch to PyObject
    let schema = batch.schema();
    let schema_fields: Vec<PyObject> = schema
        .fields()
        .iter()
        .map(|f| arrow_py.call_method1("field", (f.name(), f.data_type().to_string())).unwrap().into())
        .collect();
    let schema_py = arrow_py.call_method1("schema", (PyList::new(py, &schema_fields),))?;

    let columns: Vec<PyObject> = batch
        .columns()
        .iter()
        .map(|array| {
            match array.data_type() {
                ArrowDataType::Int32 => {
                    let arr = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                    let values: Vec<PyObject> = (0..array.len()).map(|i| {
                        if arr.is_null(i) {
                            py.None()
                        } else {
                            arr.value(i).into_py(py)
                        }
                    }).collect();
                    arrow_py.call_method1("array", (PyList::new(py, &values),)).unwrap().into_py(py)
                }
                ArrowDataType::Float64 => {
                    let arr = array.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                    let values: Vec<PyObject> = (0..array.len()).map(|i| {
                        if arr.is_null(i) {
                            py.None()
                        } else {
                            arr.value(i).into_py(py)
                        }
                    }).collect();
                    arrow_py.call_method1("array", (PyList::new(py, &values),)).unwrap().into_py(py)
                }
                ArrowDataType::Utf8 => {
                    let arr = array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                    let values: Vec<PyObject> = (0..array.len()).map(|i| {
                        if arr.is_null(i) {
                            py.None()
                        } else {
                            arr.value(i).into_py(py)
                        }
                    }).collect();
                    arrow_py.call_method1("array", (PyList::new(py, &values),)).unwrap().into_py(py)
                }
                ArrowDataType::Boolean => {
                    let arr = array.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                    let values: Vec<PyObject> = (0..array.len()).map(|i| {
                        if arr.is_null(i) {
                            py.None()
                        } else {
                            arr.value(i).into_py(py)
                        }
                    }).collect();
                    arrow_py.call_method1("array", (PyList::new(py, &values),)).unwrap().into_py(py)
                }
                _ => panic!("Unsupported data type")
            }
        })
        .collect();

    let columns_py = PyList::new(py, &columns);
    let batch_py = arrow_py.call_method1("record_batch", (columns_py, schema_py))?;
    Ok(batch_py.into())
}

#[pymodule]
fn crossbow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_excel_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv_py, m)?)?;
    Ok(())
}