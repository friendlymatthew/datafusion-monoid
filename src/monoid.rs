use std::{any::Any, marker::PhantomData, sync::Arc};

use arrow::array::{Array, ArrayRef, AsArray, ListBuilder, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field};
use arrow_schema::FieldRef;
use datafusion::common::{ScalarValue, exec_err};
use datafusion::error::Result;
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, Volatility,
    function::{AccumulatorArgs, StateFieldsArgs},
};

/// A trait for any monoid operator
///
/// It must be a binary associative operator and must contain an identity
///
/// For example, integers under addition is a monoid since:
/// - addition is a binary associative operator: (a + b = b + a)
/// - addition is commutative: (a + b) + c = a + (b + c)
/// - 0 is an identity: a + 0 = a
///
/// This trait is generic over ArrowPrimitiveType to support all numeric types
pub trait MonoidOp<T>:
    Send + Sync + std::fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + 'static
where
    T: ArrowPrimitiveType,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    fn identity() -> T::Native;
    fn combine(a: T::Native, b: T::Native) -> T::Native;
    fn name() -> &'static str;
}

use arrow::datatypes::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SumOp;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductOp;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaxOp;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MinOp;

// Macro to implement MonoidOp for integer types
macro_rules! impl_integer_monoid {
    ($op:ty, $arrow_type:ty, $native:ty, $name:expr, $identity:expr, $combine:expr) => {
        impl MonoidOp<$arrow_type> for $op {
            fn identity() -> $native {
                $identity
            }
            fn combine(a: $native, b: $native) -> $native {
                $combine(a, b)
            }
            fn name() -> &'static str {
                $name
            }
        }
    };
}

// Macro to implement MonoidOp for float types
macro_rules! impl_float_monoid {
    ($op:ty, $arrow_type:ty, $native:ty, $name:expr, $identity:expr, $combine:expr) => {
        impl MonoidOp<$arrow_type> for $op {
            fn identity() -> $native {
                $identity
            }
            fn combine(a: $native, b: $native) -> $native {
                $combine(a, b)
            }
            fn name() -> &'static str {
                $name
            }
        }
    };
}

impl_integer_monoid!(SumOp, Int8Type, i8, "sum", 0, |a: i8, b: i8| a
    .saturating_add(b));
impl_integer_monoid!(SumOp, Int16Type, i16, "sum", 0, |a: i16, b: i16| a
    .saturating_add(b));
impl_integer_monoid!(SumOp, Int32Type, i32, "sum", 0, |a: i32, b: i32| a
    .saturating_add(b));
impl_integer_monoid!(SumOp, Int64Type, i64, "sum", 0, |a: i64, b: i64| a
    .saturating_add(b));
impl_integer_monoid!(SumOp, UInt8Type, u8, "sum", 0, |a: u8, b: u8| a
    .saturating_add(b));
impl_integer_monoid!(SumOp, UInt16Type, u16, "sum", 0, |a: u16, b: u16| a
    .saturating_add(b));
impl_integer_monoid!(SumOp, UInt32Type, u32, "sum", 0, |a: u32, b: u32| a
    .saturating_add(b));
impl_integer_monoid!(SumOp, UInt64Type, u64, "sum", 0, |a: u64, b: u64| a
    .saturating_add(b));

impl_float_monoid!(SumOp, Float32Type, f32, "sum", 0.0, |a: f32, b: f32| a + b);
impl_float_monoid!(SumOp, Float64Type, f64, "sum", 0.0, |a: f64, b: f64| a + b);

impl_integer_monoid!(ProductOp, Int8Type, i8, "product", 1, |a: i8, b: i8| a
    .saturating_mul(b));
impl_integer_monoid!(ProductOp, Int16Type, i16, "product", 1, |a: i16, b: i16| a
    .saturating_mul(b));
impl_integer_monoid!(ProductOp, Int32Type, i32, "product", 1, |a: i32, b: i32| a
    .saturating_mul(b));
impl_integer_monoid!(ProductOp, Int64Type, i64, "product", 1, |a: i64, b: i64| a
    .saturating_mul(b));
impl_integer_monoid!(ProductOp, UInt8Type, u8, "product", 1, |a: u8, b: u8| a
    .saturating_mul(b));
impl_integer_monoid!(
    ProductOp,
    UInt16Type,
    u16,
    "product",
    1,
    |a: u16, b: u16| a.saturating_mul(b)
);
impl_integer_monoid!(
    ProductOp,
    UInt32Type,
    u32,
    "product",
    1,
    |a: u32, b: u32| a.saturating_mul(b)
);
impl_integer_monoid!(
    ProductOp,
    UInt64Type,
    u64,
    "product",
    1,
    |a: u64, b: u64| a.saturating_mul(b)
);

impl_float_monoid!(
    ProductOp,
    Float32Type,
    f32,
    "product",
    1.0,
    |a: f32, b: f32| a * b
);
impl_float_monoid!(
    ProductOp,
    Float64Type,
    f64,
    "product",
    1.0,
    |a: f64, b: f64| a * b
);

impl_integer_monoid!(MaxOp, Int8Type, i8, "max", i8::MIN, |a: i8, b: i8| a.max(b));
impl_integer_monoid!(MaxOp, Int16Type, i16, "max", i16::MIN, |a: i16, b: i16| a
    .max(b));
impl_integer_monoid!(MaxOp, Int32Type, i32, "max", i32::MIN, |a: i32, b: i32| a
    .max(b));
impl_integer_monoid!(MaxOp, Int64Type, i64, "max", i64::MIN, |a: i64, b: i64| a
    .max(b));
impl_integer_monoid!(MaxOp, UInt8Type, u8, "max", u8::MIN, |a: u8, b: u8| a
    .max(b));
impl_integer_monoid!(MaxOp, UInt16Type, u16, "max", u16::MIN, |a: u16, b: u16| a
    .max(b));
impl_integer_monoid!(MaxOp, UInt32Type, u32, "max", u32::MIN, |a: u32, b: u32| a
    .max(b));
impl_integer_monoid!(MaxOp, UInt64Type, u64, "max", u64::MIN, |a: u64, b: u64| a
    .max(b));

impl_float_monoid!(
    MaxOp,
    Float32Type,
    f32,
    "max",
    f32::NEG_INFINITY,
    |a: f32, b: f32| a.max(b)
);
impl_float_monoid!(
    MaxOp,
    Float64Type,
    f64,
    "max",
    f64::NEG_INFINITY,
    |a: f64, b: f64| a.max(b)
);

impl_integer_monoid!(MinOp, Int8Type, i8, "min", i8::MAX, |a: i8, b: i8| a.min(b));
impl_integer_monoid!(MinOp, Int16Type, i16, "min", i16::MAX, |a: i16, b: i16| a
    .min(b));
impl_integer_monoid!(MinOp, Int32Type, i32, "min", i32::MAX, |a: i32, b: i32| a
    .min(b));
impl_integer_monoid!(MinOp, Int64Type, i64, "min", i64::MAX, |a: i64, b: i64| a
    .min(b));
impl_integer_monoid!(MinOp, UInt8Type, u8, "min", u8::MAX, |a: u8, b: u8| a
    .min(b));
impl_integer_monoid!(MinOp, UInt16Type, u16, "min", u16::MAX, |a: u16, b: u16| a
    .min(b));
impl_integer_monoid!(MinOp, UInt32Type, u32, "min", u32::MAX, |a: u32, b: u32| a
    .min(b));
impl_integer_monoid!(MinOp, UInt64Type, u64, "min", u64::MAX, |a: u64, b: u64| a
    .min(b));

impl_float_monoid!(
    MinOp,
    Float32Type,
    f32,
    "min",
    f32::INFINITY,
    |a: f32, b: f32| a.min(b)
);
impl_float_monoid!(
    MinOp,
    Float64Type,
    f64,
    "min",
    f64::INFINITY,
    |a: f64, b: f64| a.min(b)
);

/// A generic array monoid reduce udaf
/// This udaf aggregates multiple arrays by performing element-wise reduction using a monoid operation
/// For example with the SumOp: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
#[derive(Debug, Clone)]
pub struct ArrayMonoidReduce<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    signature: Signature,
    name: String,
    _phantom_op: PhantomData<Op>,
    _phantom_type: PhantomData<T>,
}

impl<Op, T> std::cmp::PartialEq for ArrayMonoidReduce<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.signature == other.signature
    }
}

impl<Op, T> std::cmp::Eq for ArrayMonoidReduce<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
}

impl<Op, T> std::hash::Hash for ArrayMonoidReduce<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl<Op, T> ArrayMonoidReduce<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    pub fn new(data_type: DataType) -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::List(Arc::new(Field::new(
                    "item",
                    data_type.clone(),
                    true,
                )))],
                Volatility::Immutable,
            ),
            name: format!("array_reduce_{}", Op::name()),
            _phantom_op: PhantomData,
            _phantom_type: PhantomData,
        }
    }
}

impl<Op, T> AggregateUDFImpl for ArrayMonoidReduce<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        // Extract the element type from the input list type
        if let Some(DataType::List(field)) = arg_types.first() {
            Ok(DataType::List(field.clone()))
        } else {
            exec_err!("Expected List type as input")
        }
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ArrayMonoidReduceAccumulator::<Op, T>::new()))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new("array", args.return_type().clone(), true).into(),
        ])
    }

    fn groups_accumulator_supported(&self, _args: AccumulatorArgs) -> bool {
        false // Start without GroupsAccumulator for simplicity
    }
}

#[derive(Debug)]
struct ArrayMonoidReduceAccumulator<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    acc: Vec<T::Native>,
    _phantom_op: PhantomData<Op>,
    _phantom_type: PhantomData<T>,
}

impl<Op, T> ArrayMonoidReduceAccumulator<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    fn new() -> Self {
        Self {
            acc: Vec::new(),
            _phantom_op: PhantomData,
            _phantom_type: PhantomData,
        }
    }

    fn add_array(&mut self, array: &PrimitiveArray<T>) -> Result<()> {
        if self.acc.is_empty() {
            self.acc = vec![Op::identity(); array.len()];
        }

        for (j, value) in array.iter().enumerate() {
            if j >= self.acc.len() {
                // note: we can control the behavior when we get different lengths
                continue;
            }

            if let Some(val) = value {
                self.acc[j] = Op::combine(self.acc[j].clone(), val);
            }
            // note: we can also control the behavior when we get different lengths
        }

        Ok(())
    }
}

impl<Op, T> Accumulator for ArrayMonoidReduceAccumulator<Op, T>
where
    Op: MonoidOp<T>,
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
{
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        if self.acc.is_empty() {
            let mut builder = ListBuilder::new(PrimitiveArray::<T>::builder(0));
            let list_array = builder.finish();
            let null_array = list_array.slice(0, 0);
            return Ok(vec![ScalarValue::List(Arc::new(null_array))]);
        }

        let mut builder = ListBuilder::new(PrimitiveArray::<T>::builder(self.acc.len()));
        for v in &self.acc {
            builder.values().append_value(v.clone());
        }
        builder.append(true);
        let list_array = builder.finish();

        Ok(vec![ScalarValue::List(Arc::new(list_array))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.acc.is_empty() {
            let mut builder = ListBuilder::new(PrimitiveArray::<T>::builder(0));
            let list_array = builder.finish();
            let null_array = list_array.slice(0, 0);
            return Ok(ScalarValue::List(Arc::new(null_array)));
        }

        let mut builder = ListBuilder::new(PrimitiveArray::<T>::builder(self.acc.len()));
        for v in &self.acc {
            builder.values().append_value(v.clone());
        }
        builder.append(true);
        let list_array = builder.finish();

        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let [arr] = values else {
            return exec_err!("expected 1 column");
        };

        // List arrays use i32 offsets by default
        let arr = arr.as_list::<i32>();

        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }

            let array = arr.value(i);
            let primitive_array = array.as_primitive::<T>();

            self.add_array(primitive_array)?;
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let [arr] = states else {
            return exec_err!("expected 1 column");
        };

        // List arrays use i32 offsets by default
        let arr = arr.as_list::<i32>();

        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }

            let array = arr.value(i);
            let primitive_array = array.as_primitive::<T>();

            self.add_array(primitive_array)?;
        }

        Ok(())
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.acc.capacity() * std::mem::size_of::<T::Native>()
    }
}

// Factory functions for Int32 operations (backward compatibility)
pub fn array_reduce_sum() -> AggregateUDF {
    AggregateUDF::from(ArrayMonoidReduce::<SumOp, Int32Type>::new(DataType::Int32))
}

pub fn array_reduce_product() -> AggregateUDF {
    AggregateUDF::from(ArrayMonoidReduce::<ProductOp, Int32Type>::new(
        DataType::Int32,
    ))
}

pub fn array_reduce_max() -> AggregateUDF {
    AggregateUDF::from(ArrayMonoidReduce::<MaxOp, Int32Type>::new(DataType::Int32))
}

pub fn array_reduce_min() -> AggregateUDF {
    AggregateUDF::from(ArrayMonoidReduce::<MinOp, Int32Type>::new(DataType::Int32))
}

// Generic factory function for any type
pub fn array_reduce_sum_generic<T>() -> AggregateUDF
where
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
    SumOp: MonoidOp<T>,
{
    AggregateUDF::from(ArrayMonoidReduce::<SumOp, T>::new(T::DATA_TYPE))
}

pub fn array_reduce_product_generic<T>() -> AggregateUDF
where
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
    ProductOp: MonoidOp<T>,
{
    AggregateUDF::from(ArrayMonoidReduce::<ProductOp, T>::new(T::DATA_TYPE))
}

pub fn array_reduce_max_generic<T>() -> AggregateUDF
where
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
    MaxOp: MonoidOp<T>,
{
    AggregateUDF::from(ArrayMonoidReduce::<MaxOp, T>::new(T::DATA_TYPE))
}

pub fn array_reduce_min_generic<T>() -> AggregateUDF
where
    T: ArrowPrimitiveType + std::fmt::Debug + Send + Sync,
    T::Native: Clone + std::fmt::Debug + Send + Sync,
    MinOp: MonoidOp<T>,
{
    AggregateUDF::from(ArrayMonoidReduce::<MinOp, T>::new(T::DATA_TYPE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Int32Array, ListBuilder, RecordBatch},
        datatypes::Int32Type,
    };
    use arrow_schema::{Schema, SchemaRef};
    use datafusion::{catalog::MemTable, prelude::*};

    #[tokio::test]
    async fn test_array_monoid_sum_simple() -> Result<()> {
        let ctx = SessionContext::new();

        ctx.register_udaf(array_reduce_sum());

        let schema = SchemaRef::new(Schema::new(vec![Field::new(
            "arrays",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));

        let mut builder = ListBuilder::new(Int32Array::builder(0));
        builder.values().append_value(1);
        builder.values().append_value(2);
        builder.values().append_value(3);
        builder.append(true); // [1, 2, 3]

        builder.values().append_value(4);
        builder.values().append_value(5);
        builder.values().append_value(6);
        builder.append(true); // [4, 5, 6]

        let array = Arc::new(builder.finish());

        let batch = RecordBatch::try_new(schema.clone(), vec![array])?;

        let provider = MemTable::try_new(schema, vec![vec![batch]])?;
        ctx.register_table("test", Arc::new(provider))?;

        let df = ctx
            .sql("SELECT array_reduce_sum(arrays) as result FROM test")
            .await?;
        let results = df.collect().await?;

        assert_eq!(results.len(), 1);
        let result_array = results[0].column(0);
        let list_array = result_array.as_list::<i32>();

        assert!(!list_array.is_null(0));
        let result_values = list_array.value(0);
        let int_array = result_values.as_primitive::<Int32Type>();

        assert_eq!(int_array.len(), 3);
        assert_eq!(int_array.value(0), 5); // 1 + 4
        assert_eq!(int_array.value(1), 7); // 2 + 5
        assert_eq!(int_array.value(2), 9); // 3 + 6

        Ok(())
    }

    #[tokio::test]
    async fn test_array_monoid_sum_varying_lengths() -> Result<()> {
        let ctx = SessionContext::new();

        ctx.register_udaf(array_reduce_sum());

        let schema = SchemaRef::new(Schema::new(vec![Field::new(
            "arrays",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));

        let mut builder = ListBuilder::new(Int32Array::builder(0));
        builder.values().append_value(1);
        builder.values().append_value(2);
        builder.values().append_value(3);
        builder.append(true); // [1, 2, 3]

        builder.values().append_value(4);
        builder.values().append_value(5);
        builder.values().append_value(6);
        builder.values().append_value(7);
        builder.append(true); // [4, 5, 6, 7]

        let array = Arc::new(builder.finish());

        let batch = RecordBatch::try_new(schema.clone(), vec![array])?;

        let provider = MemTable::try_new(schema, vec![vec![batch]])?;
        ctx.register_table("test", Arc::new(provider))?;

        let df = ctx
            .sql("SELECT array_reduce_sum(arrays) as result FROM test")
            .await?;
        let results = df.collect().await?;

        assert_eq!(results.len(), 1);
        let result_array = results[0].column(0);
        let list_array = result_array.as_list::<i32>();

        assert!(!list_array.is_null(0));
        let result_values = list_array.value(0);
        let int_array = result_values.as_primitive::<Int32Type>();

        assert_eq!(int_array.len(), 3);
        assert_eq!(int_array.value(0), 5); // 1 + 4
        assert_eq!(int_array.value(1), 7); // 2 + 5
        assert_eq!(int_array.value(2), 9); // 3 + 6

        Ok(())
    }

    #[tokio::test]
    async fn test_array_monoid_sum_multiple_arrays() -> Result<()> {
        let ctx = SessionContext::new();

        ctx.register_udaf(array_reduce_sum());

        let schema = SchemaRef::new(Schema::new(vec![Field::new(
            "arrays",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));

        let mut builder = ListBuilder::new(Int32Array::builder(0));
        builder.values().append_value(1);
        builder.values().append_value(2);
        builder.values().append_value(3);
        builder.append(true); // [1, 2, 3]

        builder.values().append_value(4);
        builder.values().append_value(5);
        builder.values().append_value(6);
        builder.values().append_value(7);
        builder.append(true); // [4, 5, 6, 7]

        builder.values().append_value(100);
        builder.values().append_value(100);
        builder.values().append_value(100);
        builder.append(true); // [100, 100, 100] 

        let array = Arc::new(builder.finish());

        let batch = RecordBatch::try_new(schema.clone(), vec![array])?;

        let provider = MemTable::try_new(schema, vec![vec![batch]])?;
        ctx.register_table("test", Arc::new(provider))?;

        let df = ctx
            .sql("SELECT array_reduce_sum(arrays) as result FROM test")
            .await?;
        let results = df.collect().await?;

        assert_eq!(results.len(), 1);
        let result_array = results[0].column(0);
        let list_array = result_array.as_list::<i32>();

        assert!(!list_array.is_null(0));
        let result_values = list_array.value(0);
        let int_array = result_values.as_primitive::<Int32Type>();

        assert_eq!(int_array.len(), 3);
        assert_eq!(int_array.value(0), 105); // 1 + 4
        assert_eq!(int_array.value(1), 107); // 2 + 5
        assert_eq!(int_array.value(2), 109); // 3 + 6

        Ok(())
    }

    #[tokio::test]
    async fn test_array_monoid_sum_multiple_record_batches() -> Result<()> {
        let ctx = SessionContext::new();

        ctx.register_udaf(array_reduce_sum());

        let schema = SchemaRef::new(Schema::new(vec![Field::new(
            "arrays",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));

        let mut builder = ListBuilder::new(Int32Array::builder(0));
        builder.values().append_value(1);
        builder.values().append_value(2);
        builder.values().append_value(3);
        builder.append(true); // [1, 2, 3]

        builder.values().append_value(4);
        builder.values().append_value(5);
        builder.values().append_value(6);
        builder.values().append_value(7);
        builder.append(true); // [4, 5, 6, 7]

        builder.values().append_value(100);
        builder.values().append_value(100);
        builder.values().append_value(100);
        builder.append(true); // [100, 100, 100] 

        let array = Arc::new(builder.finish());

        let batch = RecordBatch::try_new(schema.clone(), vec![array])?;

        let provider = MemTable::try_new(schema, vec![vec![batch.clone(), batch.clone(), batch]])?;
        ctx.register_table("test", Arc::new(provider))?;

        let df = ctx
            .sql("SELECT array_reduce_sum(arrays) as result FROM test")
            .await?;
        let results = df.collect().await?;

        assert_eq!(results.len(), 1);
        let result_array = results[0].column(0);
        let list_array = result_array.as_list::<i32>();

        assert!(!list_array.is_null(0));
        let result_values = list_array.value(0);
        let int_array = result_values.as_primitive::<Int32Type>();

        assert_eq!(int_array.len(), 3);
        assert_eq!(int_array.value(0), 105 * 3);
        assert_eq!(int_array.value(1), 107 * 3);
        assert_eq!(int_array.value(2), 109 * 3);

        Ok(())
    }
}
