/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.cml.calcite;

import io.cml.spi.connector.Connector;
import io.cml.spi.metadata.ColumnMetadata;
import io.cml.spi.metadata.TableMetadata;
import io.cml.spi.type.PGType;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.impl.AbstractSchema;
import org.apache.calcite.sql.SqlDialect;
import org.apache.calcite.sql.dialect.BigQuerySqlDialect;
import org.apache.calcite.sql.dialect.CalciteSqlDialect;

import java.util.List;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static io.cml.spi.type.BigIntType.BIGINT;
import static io.cml.spi.type.BooleanType.BOOLEAN;
import static io.cml.spi.type.DoubleType.DOUBLE;
import static io.cml.spi.type.IntegerType.INTEGER;
import static io.cml.spi.type.VarcharType.VARCHAR;
import static org.apache.calcite.jdbc.CalciteSchema.createRootSchema;

public final class CmlSchemaUtil
{
    private CmlSchemaUtil() {}

    public enum Dialect
    {
        CALCITE(CalciteSqlDialect.DEFAULT),
        BIGQUERY(BigQuerySqlDialect.DEFAULT);

        private final SqlDialect sqlDialect;

        Dialect(SqlDialect sqlDialect)
        {
            this.sqlDialect = sqlDialect;
        }

        public SqlDialect getSqlDialect()
        {
            return sqlDialect;
        }
    }

    public static SchemaPlus schemaPlus(Connector connector)
    {
        SchemaPlus rootSchema = createRootSchema(true, true, "").plus();
        SchemaPlus secondSchema = rootSchema.add(connector.getCatalogName(), new AbstractSchema());
        connector.listSchemas()
                .forEach(schema -> secondSchema.add(schema, toCmlSchema(connector.listTables(schema))));
        return rootSchema;
    }

    private static CmlTable toCmlTable(TableMetadata tableMetadata)
    {
        JavaTypeFactoryImpl typeFactory = new JavaTypeFactoryImpl();
        RelDataTypeFactory.Builder builder = new RelDataTypeFactory.Builder(typeFactory);
        for (ColumnMetadata columnMetadata : tableMetadata.getColumns()) {
            builder.add(columnMetadata.getName(), toRelDataType(typeFactory, columnMetadata.getType()));
        }

        return new CmlTable(tableMetadata.getTable().getTableName(), builder.build());
    }

    private static CmlSchema toCmlSchema(List<TableMetadata> tables)
    {
        return new CmlSchema(tables.stream().collect(
                toImmutableMap(
                        table -> table.getTable().getTableName(),
                        CmlSchemaUtil::toCmlTable,
                        // TODO: handle case sensitive table name
                        (a, b) -> a)));
    }

    // TODO: handle nested types
    private static RelDataType toRelDataType(JavaTypeFactory typeFactory, PGType<?> pgType)
    {
        if (pgType.equals(BOOLEAN)) {
            return typeFactory.createJavaType(Boolean.class);
        }
        if (pgType.equals(INTEGER)) {
            return typeFactory.createJavaType(Integer.class);
        }
        if (pgType.equals(BIGINT)) {
            return typeFactory.createJavaType(Long.class);
        }
        if (pgType.equals(VARCHAR)) {
            return typeFactory.createJavaType(String.class);
        }
        if (pgType.equals(DOUBLE)) {
            return typeFactory.createJavaType(Double.class);
        }
        throw new UnsupportedOperationException(pgType.type() + " not supported yet");
    }
}